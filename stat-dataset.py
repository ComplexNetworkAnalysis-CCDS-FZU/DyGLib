
CSV_PATH = "./processed_data/BitcoinAlpha/ml_BitcoinAlpha.csv" 

from collections import defaultdict

import pandas as pd


def build_graph(df: pd.DataFrame):
    """构建图，保留最新边"""
    edges = {}
    for _, row in df.sort_values('ts').iterrows():
        u, v = int(row['u']), int(row['i'])
        key = (min(u,v), max(u,v))
        edges[key] = int(row['sign'])
    return edges


def count_triangles(edges: dict) -> dict:
    """统计平衡三角形"""
    adj = defaultdict(set)
    for (u, v), _ in edges.items():
        adj[u].add(v)
        adj[v].add(u)
    
    balanced = unbalanced = 0
    visited = set()
    
    for u in adj:
        for v in adj[u]:
            if v <= u: continue
            for w in adj[u] & adj[v]:
                if w <= v: continue
                tri = tuple(sorted([u, v, w]))
                if tri in visited: continue
                visited.add(tri)
                
                s1 = edges[(min(u,v), max(u,v))]
                s2 = edges[(min(v,w), max(v,w))]
                s3 = edges[(min(w,u), max(w,u))]
                
                if s1 * s2 * s3 > 0:
                    balanced += 1
                else:
                    unbalanced += 1
    
    total = balanced + unbalanced
    return {
        'triangles': total,
        'balanced': balanced,
        'balance_rate': balanced / total if total else 0
    }



df = pd.read_csv(CSV_PATH)

# 你要的5个统计
stats = {
    '节点数量': len(set(df['u']) | set(df['i'])),
    '交互数量': len(df),
    '正交互占比': (df['sign'] == 1).mean() * 100,
    '负交互占比': (df['sign'] == -1).mean() * 100,
    '持续时间(天)': (df['ts'].max() - df['ts'].min()) / 86400
}

for k, v in stats.items():
    print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

latex = f"""
\\begin{{table}}[h]
\\centering
\\caption{{数据集统计}}
\\begin{{tabular}}{{lc}}
\\toprule
指标 & 数值 \\\\
\\midrule
节点数量 & {stats['节点数量']:,} \\\\
交互数量 & {stats['交互数量']:,} \\\\
正交互占比 & {stats['正交互占比']:.2f}\\% \\\\
负交互占比 & {stats['负交互占比']:.2f}\\% \\\\
持续时间(天) & {stats['持续时间(天)']:.1f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
print(latex)


print("\n统计三角形...")
edges = build_graph(df)
tri = count_triangles(edges)
print(f"三角形总数: {tri['triangles']:,}")
print(f"平衡: {tri['balanced']:,} ({tri['balance_rate']:.2%})")
print(f"vs随机: +{tri['balance_rate']-0.5:.2%}")

# 输出LaTeX表格行
# print(f"\n{name} & {base['edges']:,} & {base['nodes']:,} & "
#       f"{base['pos_rate']:.2f}\\% & {base['neg_rate']:.2f}\\% & "
#       f"{base['duration_days']:.1f} & {tri['balance_rate']*100:.1f}\\% \\\\")