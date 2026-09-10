"""临时分析：各数据集重复交互对的「符号翻转率」（RAE 假设是否成立）与重复率。"""
import pandas as pd, numpy as np

specs = {
    "BitcoinAlpha": ("ml_BitcoinAlpha.csv", None),
    "BitcoinOTC": ("ml_BitcoinOTC.csv", None),
    "RedditHyperlinkTitle": ("ml_RedditHyperlinkTitle_tail20000.csv", 20000),
    "RedditHyperlinkBody": ("ml_RedditHyperlinkBody_tail20000.csv", 20000),
    "WikiVote": ("ml_WikiVote_tail20000.csv", 20000),
}

for name, (fn, tail) in specs.items():
    try:
        df = pd.read_csv("processed_data/" + name + "/" + fn)
    except Exception as e:
        print(name, "ERR", e)
        continue
    if tail:
        df = df.tail(tail)
    df = df.sort_values("ts")
    u = df["u"].values.astype(np.int64)
    i = df["i"].values.astype(np.int64)
    pair = np.minimum(u, i) * 1000003 + np.maximum(u, i)
    sign = df["sign"].values
    df2 = pd.DataFrame({"pair": pair, "sign": sign})
    g = df2.groupby("pair")
    sizes = g.size()
    n_rep_pairs = int((sizes >= 2).sum())
    n_rep_edges = int(sizes[sizes >= 2].sum())
    flips = trans = 0
    for _, gp in g:
        s = gp["sign"].values
        if len(s) >= 2:
            d = np.diff(s)
            trans += len(d)
            flips += int((d != 0).sum())
    print(
        f"{name}: edges={len(df)} repeat_pairs={n_rep_pairs} repeat_edges={n_rep_edges}"
        f" ({n_rep_edges/len(df)*100:.1f}%) | sign_flip_rate={flips/max(trans,1)*100:.1f}% (trans={trans})"
    )
