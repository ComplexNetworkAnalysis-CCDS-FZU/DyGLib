import datetime
import pathlib
import subprocess
import itertools
from dataclasses import dataclass
import argparse, json, sys
from typing import Iterable, List
from enum import Enum

# 只取日期
start_date = datetime.datetime.now().date()

# 日志
LOG_ROOT = pathlib.Path(f"expm-{start_date}-logs")
print(f"Save Log at {LOG_ROOT}")


class TaskTy(Enum):
    All = "all"
    Sign = "sign"
    LinkSign = "linksign"


class ExprTy(Enum):
    Ablation = "ablation"
    Hyperparameter = "parameter"


@dataclass
class Args:
    exp: list[str] = None
    name: str = ""

    def arg_list(arg_list):
        return list(itertools.chain(*[arg.exp for arg in arg_list]))

    def names(arg_list):
        return ".".join([arg.name for arg in arg_list])
    
    @classmethod
    def new(cls,*args):
        this = cls(args)
        return this


@dataclass
class ModuleCtrl:
    name: str
    enable: Args
    disable: Args

    def get_args(self, enable: bool):
        exp = (self.enable if enable else self.disable).exp
        name = self.get_name(enable)
        return Args(exp, name)

    def get_name(self, enable: bool):
        return f"{('has' if enable else 'no')}-{self.name}"

    @staticmethod
    def arg_sets(modules, group: List[List[bool]]):
        assert all(map(lambda x: len(modules) == len(x), group)), "控制长度不匹配"

        args_set = [[m.get_args(c) for m, c in zip(modules, ctrl)] for ctrl in group]

        return args_set


@dataclass
class ParamArgs:
    name: str
    arg_name: str
    enums: list[int] = None

    @classmethod
    def range(cls, name: str, arg_name: str, start: int, end: int, step: int):
        r = list(range(start, end + 1, step))
        arg = cls(name=name, arg_name=arg_name, enums=r)
        return arg

    @classmethod
    def new(cls, name: str, arg_name: str, *params):
        arg = cls(name=name, arg_name=arg_name, enums=params)
        return arg

    def args(self):
        args_set = [
            Args(
                [self.arg_name, str(en)],
                name=f"{self.name}-{en}",
            )
            for en in self.enums
        ]
        return args_set

    def with_param(self, *sepc_param):
        self.enums.extend(sepc_param)
        return self


@dataclass
class Exp:
    script: str
    dataset: str
    model: str
    extra: list[str] = None  # 每个实验私有参数
    dataset_extra: list[str] = None
    modules: list[Args] = None
    param: list[Args] = None
    gpu: int = 0,
    ablation:bool=False


# ========== 1. 参数区（可 hard-code，也可读 json） ==========
SCRIPTS = [
    "train_link_sign_prediction.py",
    "train_sign_link_3class_prediction.py",
]  # 需要跑的脚本池

SCRIPTS_TO_TASK = {
    SCRIPTS[0]:TaskTy.Sign,
    SCRIPTS[1]:TaskTy.LinkSign
}

SCRIPTS_OPTS = {
    TaskTy.All.value: SCRIPTS,
    TaskTy.Sign.value: [SCRIPTS[0]],
    TaskTy.LinkSign.value: [SCRIPTS[1]],
}


DATASETS = [
    "WikiVote",
    "BitcoinAlpha",
    "BitcoinOTC",
    "RedditHyperlinkTitle",
    "RedditHyperlinkBody",
]
MODELS = [
    "SignDyGFormer"
    #   , "DyGFormer"
]
# 任务特定参数
SCRIPT_EXTRA = {
    "train_link_sign_prediction.py": [
        "--early-stop-notice",
        "f1_binary",
        "auc",
        "f1_weighted",
    ],  # Train_A 要的
    "train_sign_link_3class_prediction.py": [
        "--early-stop-notice",
        "f1_wt",
        "f1_mic",
        "ap",
        "f1_mac",
        "auc",
    ],
}

MODULE_CTRL = [
    ModuleCtrl(
        "repeat-sampler",
        Args(["--module-repeat-aware-sampler"]),
        Args([]),
    ),
    ModuleCtrl(
        "repeat-aware-sign-encoder",
        Args(["--module-repeat-aware-sign-encoder"]),
        Args([]),
    ),
    ModuleCtrl(
        "balance-theory-encoder",
        Args([]),
        Args(["--no-module-balance-theory-encoder"]),
    ),
    ModuleCtrl(
        "module-common-neighbor-aware-sampler",
        Args([]),
        Args(
            ["--no-module-common-neighbor-aware-sampler"],
        ),
    ),
]
#
MODULE_GROUP = [
    [True, True, True, True],
    # 禁用重复感知
    [False, False, True, True],
    # 禁用平衡理论编码
    [False, False, False, True],
    # 有平衡编码，但是无共邻居采样，无重复感知
    [False, False, True, False],
    # 无共邻居采样
    [True, True, True, False],
    # 禁用全部
    [False, False, False, False],
]
PARMA_GROUPS = {
    "batch": [ParamArgs.new("batch", "--batch-size", 50, 150, 200, 250, 300)],
    "sampling": [
        ParamArgs.range("look", "--common-neighbors-look-forward", 5, 20, 5).with_param(
            1, 3
        ),
        ParamArgs.range("numN", "--num-neighbors", 20, 100, 20).with_param(10, 15),
    ],
}
PARMA_CTRL = [
    ParamArgs.range("look", "--common-neighbors-look-forward", 5, 20, 5).with_param(
        1, 3
    ),
    ParamArgs.range("numN", "--num-neighbors", 20, 100, 20).with_param(10, 15),
]

DEFAULT_PARMA = [
    Args(["--common-neighbors-look-forward", "5"]),
    Args(["--num-neighbors", "20"]),
]

# 公共参数
COMMA_EXTRA = ["--num-runs", "1"]
# 如果实验太多，把上面内容写 experiments.json 然后 json.load 即可
DATASET_EXTRA = {
    "RedditHyperlinkBody": ["--tail-num", "20000"],
    "RedditHyperlinkTitle": ["--tail-num", "20000"],
}


TASK_DATASET_BEST_PARAMS = {
    TaskTy.Sign.value: {
        "WikiVote": Args(["--batch-size", "200"]),
        "BitcoinAlpha": Args(["--batch-size", "200"]),
        "BitcoinOTC": Args(["--batch-size", "200"]),
        "RedditHyperlinkTitle": Args(["--batch-size", "200"]),
        "RedditHyperlinkBody": Args(["--batch-size", "200"]),
    },
    TaskTy.Sign.value: {
        "WikiVote": Args(["--batch-size", "200"]),
        "BitcoinAlpha": Args(["--batch-size", "200"]),
        "BitcoinOTC": Args(["--batch-size", "200"]),
        "RedditHyperlinkTitle": Args(["--batch-size", "200"]),
        "RedditHyperlinkBody": Args(["--batch-size", "200"]),
    },
}


# ========== 2. 生成笛卡尔积 ==========、
def make_experiments(
    *,
    param_expm: bool = False,
    gpu: int = 0,
    script_type: str = "all",
    skip_dataset: list = None,
    param_ty: str = "sampling",
):
    ablation = ModuleCtrl.arg_sets(
            MODULE_CTRL, [[True, True, True, True]] if param_expm else MODULE_GROUP
        ),
    params_fn =lambda dataset,script: [TASK_DATASET_BEST_PARAMS[SCRIPTS_TO_TASK[script].value][dataset]]

    for item in itertools.product(
        SCRIPTS_OPTS[script_type],
        DATASETS,
        MODELS,
        ablation,
        *(
            [p.args() for p in PARMA_GROUPS[param_ty]]
            if param_expm
            else [[DEFAULT_PARMA]]
        ),
    ):
        s = item[0]
        d = item[1]
        m = item[2]
        mc = item[3]
        p = item[4:] if len(item) > 4 else []

        if skip_dataset is not None and d in skip_dataset:
            print(f"skip dataset {d}")
            continue
        yield Exp(
            script=(s),
            dataset=d,
            model=m,
            extra=SCRIPT_EXTRA.get(s),
            modules=mc,
            param=p if param_expm else params_fn(d,s),
            dataset_extra=DATASET_EXTRA.get(d, []),
            gpu=gpu,
            ablation=not param_expm
        )


# ========== 3. 顺序运行 ==========
def run(exp: Exp, dry_run: bool = False):
    cmd = [
        sys.executable,
        exp.script,
        "--dataset-name",
        exp.dataset,
        "--model",
        exp.model,
        "--gpu",
        str(exp.gpu),
        *["--ablation"] if exp.ablation else [],
        *exp.extra,
        *exp.dataset_extra,
        *COMMA_EXTRA,
        *Args.arg_list(exp.modules),
        *Args.arg_list(exp.param),
    ]
    log_dir = LOG_ROOT / pathlib.Path(exp.script).stem
    log_dir.mkdir(parents=True, exist_ok=True)

    # 3. 日志文件：Dataset_Model_时间戳.log
    ts = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_file = (
        log_dir
        / f"{exp.dataset}_{exp.model}.{Args.names(exp.modules)}.{Args.names(exp.param)}_{ts}.log"
    )

    print(f"Save log at : {log_file}")

    if dry_run:
        return " ".join(cmd)
    print(">>>", " ".join(cmd))
    with log_file.open("w", encoding="utf-8") as f:
        ret = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if ret.returncode != 0:
        print(f"[ERROR] 实验失败: {exp}", file=sys.stderr)
    print("[logged →", log_file, "]")


# ========== 4. CLI ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--dry-run", action="store_true", help="只打印命令不真跑")
    ap.add_argument(
        "-p", "--param-expm", action="store_true", help="开启全部模块，进行精度实验"
    )
    ap.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=0,
    )
    ap.add_argument(
        "-s",
        "--script",
        type=str,
        choices=list(TaskTy.__members__.keys()),
        default="all",
    )
    ap.add_argument("-r", "--ignore-dataset", nargs="+", required=False, default=None)
    ap.add_argument(
        "-t",
        "--param-type",
        default="sampling",
        choices=PARMA_GROUPS.keys(),
        help="参数实验类型",
    )
    args = ap.parse_args()

    for exp in make_experiments(
        param_expm=args.param_expm,
        gpu=args.gpu,
        script_type=args.script,
        skip_dataset=args.ignore_dataset,
        param_ty=args.param_type,
    ):
        if args.dry_run:
            print(run(exp, dry_run=True))
        else:
            run(exp)


if __name__ == "__main__":
    main()
