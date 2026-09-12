"""tools/sync/fetch_results.py — 把服务器上已完成的实验结果 JSON 同步到本地归档。

用法（仓库根目录）：
    python tools/sync/fetch_results.py --set e3      # E-3 修复后 8 个（RB+WV × P{1,3,5,7}）
    python tools/sync/fetch_results.py --set e4      # E-4 修复后 TD ×1（WV NN-15 LF-10）
    python tools/sync/fetch_results.py --set main-a  # 主表第一批：sign RT+RB（10）
    python tools/sync/fetch_results.py --set e2      # E-2 修复后 20 个（待队列 #14/#15 完成后）
    python tools/sync/fetch_results.py --set e5      # E-5 10 个（sign RT 已可；linksign RT 待完整）
    python tools/sync/fetch_results.py --set all     # e2 + e5

纪律（2026-09-11 用户指示）：
- **只读服务器**（ssh 读取；不写/不删/不 scp）；服务器访问须**用户逐次明确许可**。
- 本脚本把「服务器 → 本地 results/**/raw/」的拉取动作固化为可追溯流程：
  单次 ssh 调用（服务器侧 python 打包 base64），本地写入后打印 sha256 清单，
  并向 `results/_sync_raw_log.csv` 追加一行（时间/文件/sha256）。
- 拉取后可用 `python tools/verify/check_archived_results.py` 对照本地归档指纹。
"""
from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # 仓库根（tools/sync/ 的上两级）
SSH_TARGET = "fedsa@172.17.173.102"
REMOTE_CMD = (
    "cd ~/DyGLib && source /home/fedsa/anaconda3/etc/profile.d/conda.sh "
    "&& conda activate gc && python -"
)

# 每个条目: (remote_glob 模板, local_dir 模板, 数据集列表 or None, 期望文件数)
E2_BASE = "saved_results/SignLinkPrediction/SignDyGFormer/{ds}/*NN-Best.LF-Best.RAS-?.RASE-?.BTE-E.CNAS-E.P1.TE.json"
E5_LINKSIGN = "saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkTitle/*NN-60.LF-1.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
E5_SIGN = "saved_results/LinkSign/SignDyGFormer/RedditHyperlinkTitle/*NN-100.LF-1.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
E3_RB = "saved_results/SignLinkPrediction/SignDyGFormer/RedditHyperlinkBody/*NN-80.LF-3.RAS-E.RASE-E.BTE-E.CNAS-E.P[1357].TE.json"
E3_WV = "saved_results/SignLinkPrediction/SignDyGFormer/WikiVote/*NN-15.LF-10.RAS-E.RASE-E.BTE-E.CNAS-E.P[1357].TE.json"
E4_TD = "saved_results/SignLinkPrediction/SignDyGFormer/WikiVote/*NN-15.LF-10.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TD.json"
MAIN_SIGN_RT = "saved_results/LinkSign/SignDyGFormer/RedditHyperlinkTitle/*NN-100.LF-1.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"
MAIN_SIGN_RB = "saved_results/LinkSign/SignDyGFormer/RedditHyperlinkBody/*NN-60.LF-1.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json"

SETS = {
    "e2": [
        (E2_BASE, "results/E-2_ablation/raw/{ds}",
         ["BitcoinAlpha", "BitcoinOTC", "WikiVote", "RedditHyperlinkTitle", "RedditHyperlinkBody"], 20),
    ],
    "e3": [
        (E3_RB, "results/E-3_patch/raw", None, 4),
        (E3_WV, "results/E-3_patch/raw", None, 4),
    ],
    "e4": [
        (E4_TD, "results/E-4_time_decay/raw", None, 1),
    ],
    "e5": [
        (E5_LINKSIGN, "results/E-5_significance/raw/linksign", None, 5),
        (E5_SIGN, "results/E-5_significance/raw/sign", None, 5),
    ],
    "main-a": [
        (MAIN_SIGN_RT, "results/main_tables/raw/sign/RedditHyperlinkTitle", None, 5),
        (MAIN_SIGN_RB, "results/main_tables/raw/sign/RedditHyperlinkBody", None, 5),
    ],
}
SETS["all"] = SETS["e2"] + SETS["e5"]


def build_globs(specs):
    """展开成 [(glob, local_dir)] 列表。"""
    out = []
    for (gt, lt, datasets, _exp) in specs:
        if datasets:
            for ds in datasets:
                out.append((gt.format(ds=ds), lt.format(ds=ds)))
        else:
            out.append((gt, lt))
    return out


def remote_fetch(globs) -> list:
    """单次 ssh 拉取：返回 [[glob_idx, remote_path, b64], ...]。"""
    code = (
        "import glob, base64, json\n"
        f"globs = {json.dumps([g for g, _ in globs])}\n"
        "out = []\n"
        "for gi, g in enumerate(globs):\n"
        "    for f in sorted(glob.glob(g)):\n"
        "        out.append([gi, f, base64.b64encode(open(f, 'rb').read()).decode()])\n"
        "print('===SYNC_JSON===')\n"
        "print(json.dumps(out))\n"
        "print('===SYNC_END===')\n"
    )
    res = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=15", SSH_TARGET, REMOTE_CMD],
        input=code, capture_output=True, text=True, encoding="utf-8", timeout=300,
    )
    if "===SYNC_JSON===" not in (res.stdout or ""):
        print("[ERR] 服务器无预期输出；stderr 尾部：", file=sys.stderr)
        print((res.stderr or "")[-800:], file=sys.stderr)
        sys.exit(1)
    body = res.stdout.split("===SYNC_JSON===")[1].split("===SYNC_END===")[0].strip()
    return json.loads(body)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(SETS), required=True)
    args = ap.parse_args()

    specs = SETS[args.set]
    globs = build_globs(specs)
    expect = sum(e for *_, e in specs)

    print(f"[fetch] set={args.set} 期望 {expect} 个文件；ssh 读取中…")
    items = remote_fetch(globs)
    if len(items) != expect:
        print(f"[ERR] 实际匹配 {len(items)} 个，期望 {expect} —— 模式可能与服务器不符，中止。")
        sys.exit(1)

    log_path = ROOT / "results" / "_sync_raw_log.csv"
    new_log = not log_path.exists()
    log_f = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(log_f)
    if new_log:
        writer.writerow(["time", "set", "remote", "local", "sha256", "bytes"])

    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{'数据集/目录':<22s} {'文件':<58s} {'KB':>7s}  sha256[:12]")
    print("-" * 110)
    for gi, rpath, b64 in items:
        _, ldir = globs[gi]
        data = base64.b64decode(b64)
        local = ROOT / ldir / Path(rpath).name
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        dig = sha256_bytes(data)
        writer.writerow([ts, args.set, rpath, str(local.relative_to(ROOT)).replace("\\", "/"), dig, len(data)])
        try:
            auc = json.loads(data.decode("utf-8"))["test metrics"]["auc"]
        except Exception:
            auc = "-"
        print(f"{ldir.split('/')[-1]:<22s} {Path(rpath).name:<58s} {len(data)/1024:7.1f}  {dig[:12]}  auc={auc}")
    log_f.close()
    print(f"[ok] 已写入 {len(items)} 个文件；日志追加于 {log_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
