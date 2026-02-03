import datetime
import pathlib
import subprocess
import itertools
from dataclasses import dataclass
import argparse, json, sys

# 日志
LOG_ROOT = pathlib.Path("expm-logs")


@dataclass
class Exp:
    script: str
    dataset: str
    model: str
    extra: list[str] = None  # 每个实验私有参数
    dataset_extra: list[str] = None
    modules: list[str] = None


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
MODELS = ["SignDyGFormer"
        #   , "DyGFormer"
          ]
# 任务特定参数
SCRIPT_EXTRA = {
    "train_link_sign_prediction.py": [
        "--early-stop-notice",
        "f1_binary",
        "auc",
        "acc",
    ],  # Train_A 要的
    "train_sign_link_3class_prediction.py": [
        "--early-stop-notice",
        "exist_recall",
        "sign_f1",
        "auc",
    ],  # Train_B 要的
}

MODULE_CTRL = [["--repeat-aware"], []]

# 公共参数
COMMA_EXTRA = ["--num-runs", "3"]
# 如果实验太多，把上面内容写 experiments.json 然后 json.load 即可
DATASET_EXTRA = {
    "RedditHyperlinkBody": ["--tail-num", "20000"],
    "RedditHyperlinkTitle": ["--tail-num", "20000"],
}


# ========== 2. 生成笛卡尔积 ==========
def make_experiments():
    for s, d, m, mc in itertools.product(SCRIPTS, DATASETS, MODELS, MODULE_CTRL):
        yield Exp(
            script=(s),
            dataset=d,
            model=m,
            extra=SCRIPT_EXTRA.get(s),
            modules=mc,
            dataset_extra=DATASET_EXTRA.get(d, []),
        )


# ========== 3. 顺序运行 ==========
def run(exp: Exp):
    extra = SCRIPT_EXTRA.get(exp.script, [])
    cmd = [
        sys.executable,
        exp.script,
        "--dataset-name",
        exp.dataset,
        "--model",
        exp.model,
        *exp.extra,
        *exp.modules,
        *exp.dataset_extra,
        *COMMA_EXTRA,
    ]

    log_dir = LOG_ROOT / pathlib.Path(exp.script).stem
    log_dir.mkdir(parents=True, exist_ok=True)

    # 3. 日志文件：Dataset_Model_时间戳.log
    ts = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_file = log_dir / f"{exp.dataset}_{exp.model}_{ts}.log"

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
            print(
                " ".join(
                    [
                        sys.executable,
                        exp.script,
                        "--dataset-name",
                        exp.dataset,
                        "--model",
                        exp.model,
                        *exp.extra,
                        *exp.modules,
                        *exp.dataset_extra,
                        *COMMA_EXTRA,
                    ]
                )
            )
        else:
            run(exp)


if __name__ == "__main__":
    main()
