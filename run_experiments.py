import datetime
import pathlib
import subprocess
import itertools
from dataclasses import dataclass
import argparse, json, sys


# 只取日期
start_date = datetime.datetime.now().date()

# 日志
LOG_ROOT = pathlib.Path(f"expm-{start_date}-logs")
print(f"Save Log at {LOG_ROOT}")


@dataclass
class ModuleCtrl:
    name: str
    exp: list[str] = None


@dataclass
class Exp:
    script: str
    dataset: str
    model: str
    extra: list[str] = None  # 每个实验私有参数
    dataset_extra: list[str] = None
    modules: ModuleCtrl = None


# ========== 1. 参数区（可 hard-code，也可读 json） ==========
SCRIPTS = [
    "train_link_sign_prediction.py",
    "train_sign_link_3class_prediction.py",
]  # 需要跑的脚本池
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

MODULE_CTRL = {
    "module_repeat_aware_sampler": [
        ModuleCtrl("has-repeat-sampler", ["--module-repeat-aware-sampler"]),
        ModuleCtrl("no-repeat-sampler", []),
    ],
    "module_repeat_aware_sign_encoder": [
        ModuleCtrl(
            "has-repeat-aware-sign_encoder", ["--module-repeat-aware-sign-encoder"]
        ),
        ModuleCtrl("no-repeat-aware-sign_encoder", []),
    ],
    "module_balance_theory_encoder": [
        ModuleCtrl("has-balance-theory-encoder", []),
        ModuleCtrl("no-balance-theory-encoder", ["--no-module-balance-theory-encoder"]),
    ],
    "module_common_neighbor_aware_sampler": [
        ModuleCtrl("has-module-common-neighbor-aware-sampler", []),
        ModuleCtrl(
            "no-module-common-neighbor-aware-sampler",
            ["--no-module-common-neighbor-aware-sampler"],
        ),
    ],
    # "common-neighbors-look-forward":[
    #     ModuleCtrl("look1",["--common-neighbors-look-forward","1"]),
    #     ModuleCtrl("look5",["--common-neighbors-look-forward","5"]),
    #     ModuleCtrl("look10",["--common-neighbors-look-forward","10"]),
    #     ModuleCtrl("look20",["--common-neighbors-look-forward","20"]),
    # ],
    # "num-neighbor":[
    #     ModuleCtrl("num10",["--num-neighbors","10"]),
    #     ModuleCtrl("num20",["--num-neighbors","20"]),
    #     ModuleCtrl("num32",["--num-neighbors","32"]),
    #     ModuleCtrl("num64",["--num-neighbors","64"]),
    #     ModuleCtrl("num100",["--num-neighbors","100"]),
    # ]
}

# 公共参数
COMMA_EXTRA = ["--num-runs", "3"]
# 如果实验太多，把上面内容写 experiments.json 然后 json.load 即可
DATASET_EXTRA = {
    "RedditHyperlinkBody": ["--tail-num", "20000"],
    "RedditHyperlinkTitle": ["--tail-num", "20000"],
}


# ========== 2. 生成笛卡尔积 ==========、
def make_experiments():
    for item in itertools.product(SCRIPTS, DATASETS, MODELS, *MODULE_CTRL.values()):
        s = item[0]
        d = item[1]
        m = item[2]
        mc = [] if len(item) == 3 else item[3:]
        yield Exp(
            script=(s),
            dataset=d,
            model=m,
            extra=SCRIPT_EXTRA.get(s),
            modules=mc,
            dataset_extra=DATASET_EXTRA.get(d, []),
        )


# ========== 3. 顺序运行 ==========
def run(exp: Exp, dry_run: bool = False):
    extra = SCRIPT_EXTRA.get(exp.script, [])
    cmd = [
        sys.executable,
        exp.script,
        "--dataset-name",
        exp.dataset,
        "--model",
        exp.model,
        *exp.extra,
        *list(itertools.chain(*[modules.exp for modules in exp.modules])),
        *exp.dataset_extra,
        *COMMA_EXTRA,
    ]
    if dry_run:
        return " ".join(cmd)
    log_dir = LOG_ROOT / pathlib.Path(exp.script).stem
    log_dir.mkdir(parents=True, exist_ok=True)

    # 3. 日志文件：Dataset_Model_时间戳.log
    ts = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_file = (
        log_dir
        / f"{exp.dataset}_{exp.model}_{'_'.join([m.name for m in exp.modules])}_{ts}.log"
    )

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
    args = ap.parse_args()

    for exp in make_experiments():
        if args.dry_run:
            print(run(exp, dry_run=True))
        else:
            run(exp)


if __name__ == "__main__":
    main()
