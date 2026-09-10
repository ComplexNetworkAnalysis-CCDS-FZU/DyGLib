"""读取本地存档的 E-2 原始结果（修复前拷贝），打印时间/指标，用于与服务器新文件对照。"""
import json, glob, os, time

for ds in ["BitcoinAlpha", "RedditHyperlinkTitle"]:
    for f in sorted(glob.glob("results/E-2_ablation/raw/" + ds + "/*.json")):
        d = json.load(open(f))
        print(
            ds,
            os.path.basename(f).split("RAS-")[1][:15],
            "time=%.0f" % d["single run time (s)"],
            "mtime=" + time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(f))),
            "auc=" + d["test metrics"]["auc"],
        )
