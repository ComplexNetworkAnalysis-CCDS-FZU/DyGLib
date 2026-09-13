import datetime
import pathlib
import subprocess
import itertools
from dataclasses import dataclass
import argparse, json, sys
from typing import Iterable, List
from enum import Enum
import functools

# 只取日期
start_date = datetime.datetime.now().date()

# 日志
LOG_ROOT = pathlib.Path(f"expm-{start_date}-logs")
print(f"Save Log at {LOG_ROOT}")


class TaskTy(Enum):
    All = "all"
    Sign = "sign"
    LinkSign = "linksign"
    DirectLink = "directlink"


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
    def new(cls, *args):
        this = cls(args)
        return this


class ParamArgs(Args):
    param_name: str
    param: str


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
            Args([self.arg_name, str(en)], name=f"{self.name}-{en}",)
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
    gpu: int = (0,)
    ablation: bool = False,
    seeds:bool = False,
    accel: bool = None  # M4：加速开关（None=子进程默认；True/False=显式透传）


from models.DyGFormer import DyGFormer
from models.SignDyGFormer import SignDyGFormer
from models.DirectSignDyGFormer import DirectSignDyGFormer

# ========== 1. 参数区（可 hard-code，也可读 json） ==========
SCRIPTS = [
    "train_link_sign_prediction.py",
    "train_sign_link_3class_prediction.py",
    "train_direct_link_prediction.py",
]  # 需要跑的脚本池

SCRIPTS_TO_TASK = {
    SCRIPTS[0]: TaskTy.Sign,
    SCRIPTS[1]: TaskTy.LinkSign,
    SCRIPTS[2]: TaskTy.DirectLink,
}

SCRIPTS_OPTS = {
    TaskTy.All.value: SCRIPTS,
    TaskTy.Sign.value: [SCRIPTS[0]],
    TaskTy.LinkSign.value: [SCRIPTS[1]],
    TaskTy.DirectLink.value: [SCRIPTS[2]],
}


DATASETS = [
    "BitcoinAlpha",
    "BitcoinOTC",
    "RedditHyperlinkTitle",
    "RedditHyperlinkBody",
    "WikiVote",
]
MODELS = [
    SignDyGFormer.NAME,
    DirectSignDyGFormer.NAME,
    DyGFormer.NAME,
    "TGAT",
    "GraphMixer",
]
# 任务特定参数
SCRIPT_EXTRA = {
    "train_link_sign_prediction.py": [
        "--early-stop-notice",
        "f1_binary",
        "auc",
        "f1_weighted",
    ],
    "train_sign_link_3class_prediction.py": [
        "--early-stop-notice",
        "f1_wt",
        "f1_mic",
        "ap",
        "f1_mac",
        "auc",
    ],
    "train_direct_link_prediction.py": [
        "--early-stop-notice",
        "ap",
        "auc",
        "f1_binary",
    ],
}

MODULE_CTRL = [
    ModuleCtrl("repeat-sampler", Args(["--module-repeat-aware-sampler"]), Args([]),),
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
        Args(["--no-module-common-neighbor-aware-sampler"],),
    ),
]
#
MODULE_GROUP = [
    # ===== E-2 消融（导师方案 R1-5）：CNAS+BTE 固定基座，RAS/RAE 解绑 =====
    # 顺序: [RAS, RAE, BTE, CNAS]
    # 1. 基座 CNAS+BTE
    [False, False, True, True],
    # 2. +RAS
    [True, False, True, True],
    # 3. +RAE
    [False, True, True, True],
    # 4. 全开（完整模型）
    [True, True, True, True],
    # ===== E-2b 补充（Paper 09-13；BTE 关断对照）=====
    # 5. CNAS-only（BTE 关）
    [False, False, False, True],
    # 6. vanilla（全关）
    [False, False, False, False],
]

class ExpTy(Enum):
    Main= "main"
    Ablation = "ablation"
    Hyperparameter = "parameter"
    Patch = "patch"

PARMA_GROUPS = {
    ExpTy.Hyperparameter.value: [
        ParamArgs.range("look", "--common-neighbors-look-forward", 5, 20, 5).with_param(
            1, 3
        ),
        ParamArgs.range("numN", "--num-neighbors", 20, 100, 20).with_param(10, 15),
    ],
    ExpTy.Main.value: [],
    ExpTy.Ablation.value: [],
    # E-3: patch size 消融
    ExpTy.Patch.value: [
        ParamArgs.new("P", "--patch-size", 1, 3, 5, 7),
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
COMMA_EXTRA = []
# 如果实验太多，把上面内容写 experiments.json 然后 json.load 即可
DATASET_EXTRA = {
    "RedditHyperlinkBody": ["--tail-num", "20000"],
    "RedditHyperlinkTitle": ["--tail-num", "20000"],
    # 修订实验统一 top20000 简化（E-1~E-5）
    "WikiVote": ["--tail-num", "20000"],
}


TASK_DATASET_BEST_PARAMS = {
    TaskTy.Sign.value: {
        "WikiVote": [
            Args(["--batch-size", "200"]),
            Args.new("--num-neighbors", "40"),
            Args.new("--common-neighbors-look-forward", "15"),
        ],
        "BitcoinAlpha": [
            Args(["--batch-size", "200"]),
            Args.new("--num-neighbors", "40"),
            Args.new("--common-neighbors-look-forward", "15"),
        ],
        "BitcoinOTC": [
            Args(["--batch-size", "200"]),
            Args.new("--num-neighbors", "60"),
            Args.new("--common-neighbors-look-forward", "10"),
        ],
        "RedditHyperlinkTitle": [
            Args(["--batch-size", "200"]),
            Args.new("--num-neighbors", "100"),
            Args.new("--common-neighbors-look-forward", "1"),
        ],
        "RedditHyperlinkBody": [
            Args(["--batch-size", "200"]),
            Args.new("--num-neighbors", "60"),
            Args.new("--common-neighbors-look-forward", "1"),
        ],
    },
    TaskTy.LinkSign.value: {
        "WikiVote": [
            Args(["--batch-size", "200"]),
            Args.new("--num-neighbors", "15"),
            Args.new("--common-neighbors-look-forward", "10"),
        ],
        "BitcoinAlpha": [
            Args(["--batch-size", "200"]),
            Args.new("--num-neighbors", "40"),
            Args.new("--common-neighbors-look-forward", "15"),
        ],
        "BitcoinOTC": [
            Args(["--batch-size", "200"]),
            Args.new("--num-neighbors", "80"),
            Args.new("--common-neighbors-look-forward", "5"),
        ],
        "RedditHyperlinkTitle": [
            Args(["--batch-size", "200"]),
            Args.new("--num-neighbors", "60"),
            Args.new("--common-neighbors-look-forward", "1"),
        ],
        "RedditHyperlinkBody": [
            Args(["--batch-size", "200"]),
            Args.new("--num-neighbors", "80"),
            Args.new("--common-neighbors-look-forward", "3"),
        ],
    },
    TaskTy.DirectLink.value: {
        "WikiVote": [],
        "BitcoinAlpha": [],
        "BitcoinOTC": [],
        "RedditHyperlinkTitle": [],
        "RedditHyperlinkBody": [],
    },
}

TASK_SEED_VALUES = [42, 123, 456, 789, 1024]

# ========== 2. 生成笛卡尔积 ==========、
def make_experiments(
    *,
    gpu: int = 0,
    script_type: str = "all",
    skip_dataset: list = None,
    models: list = None,
    exp_ty: ExpTy = ExpTy.Main,
    seed_expm: bool = False,
    module_idx: list = None,
    accel: bool = None,
):
    if exp_ty == ExpTy.Ablation:
        group = (
            MODULE_GROUP
            if module_idx is None
            else [MODULE_GROUP[i] for i in module_idx]
        )
        ablation = ModuleCtrl.arg_sets(MODULE_CTRL, group)
    else:
        ablation = ModuleCtrl.arg_sets(MODULE_CTRL, [[True, True, True, True]])

    params_fn = lambda dataset, script: TASK_DATASET_BEST_PARAMS[
        SCRIPTS_TO_TASK[script].value
    ][dataset]

    models_pool = MODELS if models is None else models

    for item in itertools.product(
        SCRIPTS_OPTS[script_type],
        DATASETS,
        models_pool,
        ablation,
        *(
            [p.args() for p in PARMA_GROUPS[exp_ty.value]]
            if exp_ty in (ExpTy.Hyperparameter, ExpTy.Patch)
            else [[DEFAULT_PARMA]]
        ),
    ):
        s = item[0]
        d = item[1]
        m = item[2]
        mc = item[3]
        p = item[4:] if len(item) > 4 else []

        # 非 SignDyGFormer 家族的模型不支持模块消融
        if m not in (SignDyGFormer.NAME, DirectSignDyGFormer.NAME):
            mc = []

        if skip_dataset is not None and d in skip_dataset:
            print(f"skip dataset {d}")
            continue
        if exp_ty == ExpTy.Hyperparameter:
            param = p
        elif exp_ty == ExpTy.Patch:
            # E-3: patch size 扫描 + 其余超参沿用最佳配置
            param = list(p) + params_fn(d, s)
        else:
            param = params_fn(d, s)
        yield Exp(
            script=(s),
            dataset=d,
            model=m,
            extra=SCRIPT_EXTRA.get(s),
            modules=mc,
            param=param,
            dataset_extra=DATASET_EXTRA.get(d, []),
            gpu=gpu,
            ablation=exp_ty == ExpTy.Ablation,
            seeds=seed_expm,
            accel=accel,
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
        "--seeds",
        *([str(TASK_SEED_VALUES[0])] if not exp.seeds else [str(s) for s in TASK_SEED_VALUES]),
        *(["--ablation"] if exp.ablation else []),
        *(["--accel"] if exp.accel is True else (["--no-accel"] if exp.accel is False else [])),
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
    ap.add_argument("-p", "--param-expm", action="store_true", help="开启全部模块，进行精度实验")
    ap.add_argument(
        "-g", "--gpu", type=int, default=0,
    )
    ap.add_argument(
        "-s",
        "--script",
        type=str,
        choices=list([v.value for v in TaskTy.__members__.values()]),
        default="all",
    )
    ap.add_argument("-r", "--ignore-dataset", nargs="+", required=False, default=None)
    ap.add_argument(
        "-m",
        "--models",
        nargs="+",
        required=False,
        default=None,
        help="只运行指定模型（默认全部）",
    )
    ap.add_argument(
        "-t",
        "--exp-type",
        default=ExpTy.Main.value,
        choices=list([v.value for v in ExpTy.__members__.values()]),
        help="参数实验类型",
    )
    ap.add_argument(
        "--module-idx",
        nargs="+",
        type=int,
        required=False,
        default=None,
        help="消融模式只运行指定 MODULE_GROUP 行号（0-based），默认全部",
    )
    ap.add_argument("-e", "--seed_expm", action="store_true", help="开启随机种子实验")
    ap.add_argument(
        "--accel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="加速后端开关（默认启用；--no-accel 关闭）",
    )

    args = ap.parse_args()

    for exp in make_experiments(
        gpu=args.gpu,
        script_type=args.script,
        skip_dataset=args.ignore_dataset,
        models=args.models,
        exp_ty=ExpTy(args.exp_type),
        seed_expm=args.seed_expm,
        module_idx=args.module_idx,
        accel=args.accel,
    ):
        if args.dry_run:
            print(run(exp, dry_run=True))
        else:
            run(exp)


if __name__ == "__main__":
    main()