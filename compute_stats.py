"""
E-5 统计显著性计算脚本（SignDyG 修订实验）
================================================
- 读取 saved_results 下各随机种子的结果 JSON
- 输出 mean ± std（跨种子）
- 两个配置 / 两个模型之间做配对 t 检验（scipy.stats.ttest_rel），输出 p 值

结果 JSON 路径约定（由训练脚本写入）:
  sign     -> ./saved_results/LinkSign/{model}/{dataset}/{result_save_name}.json
  linksign -> ./saved_results/SignLinkPrediction/{model}/{dataset}/{result_save_name}.json
  direct   -> ./saved_results/DirectLinkPred/{model}/{dataset}/{result_save_name}.json

result_save_name 形如:
  {Model}_seed{seed}.NN-Best.LF-Best.RAS-E.RASE-E.BTE-E.CNAS-E
  （E=启用, D=禁用; NN/LF 为超参或 Best）

用法示例:
  # 1) 汇总完整模型在 RedditTitle 上的 5 种子结果
  python compute_stats.py --task linksign --dataset RedditHyperlinkTitle --model SignDyGFormer \
      --pattern "RAS-E.RASE-E.BTE-E.CNAS-E"

  # 2) 完整模型 vs 消融基线（同模型不同配置），输出配对 t 检验 p 值
  python compute_stats.py --task linksign --dataset RedditHyperlinkTitle --model SignDyGFormer \
      --pattern "RAS-E.RASE-E.BTE-E.CNAS-E" --compare "RAS-D.RASE-D.BTE-D.CNAS-D"

  # 3) 完整模型 vs 基线模型（如 DyGFormer），输出配对 t 检验 p 值
  python compute_stats.py --task linksign --dataset RedditHyperlinkTitle --model SignDyGFormer \
      --pattern "RAS-E.RASE-E.BTE-E.CNAS-E" --compare-model DyGFormer

  # 4) 只汇总不比较，并输出 CSV
  python compute_stats.py --task sign --dataset RedditHyperlinkTitle --model SignDyGFormer \
      --pattern "RAS-E.RASE-E.BTE-E.CNAS-E" --output stats_summary.csv
"""

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

TASK_DIRS = {
    "sign": "LinkSign",
    "linksign": "SignLinkPrediction",
    "direct": "DirectLinkPred",
}

SEED_RE = re.compile(r"_seed(\d+)")


def load_seed_results(folder: str, pattern: str) -> Dict[int, Dict[str, float]]:
    """
    加载 folder 下所有 result_save_name 含 pattern 的 JSON（按种子聚合）。
    :return: {seed: {metric_name: value}}
    """
    files = glob.glob(os.path.join(folder, f"*{pattern}*.json"))
    results: Dict[int, Dict[str, float]] = {}
    for fp in files:
        m = SEED_RE.search(os.path.basename(fp))
        if m is None:
            print(f"[warn] 无法从文件名提取种子，跳过: {fp}")
            continue
        seed = int(m.group(1))
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        results[seed] = data
    return results


def extract_metrics(data: Dict, prefix: str = "test ") -> Dict[str, float]:
    """提取所有以 prefix 开头的指标。"""
    out = {}
    for k, v in data.items():
        if isinstance(k, str) and k.startswith(prefix) and isinstance(v, (int, float)):
            out[k[len(prefix):]] = float(v)
    return out


def summarize(seed_metrics: Dict[int, Dict[str, float]]) -> Dict[str, Tuple[float, float, int]]:
    """对每个指标计算 mean ± std（ddof=1）。"""
    summary: Dict[str, Tuple[float, float, int]] = {}
    # 取并集指标
    metric_names = set()
    for v in seed_metrics.values():
        metric_names.update(extract_metrics(v).keys())
    for name in sorted(metric_names):
        vals = np.array(
            [extract_metrics(v)[name] for v in seed_metrics.values() if name in extract_metrics(v)],
            dtype=float,
        )
        if len(vals) == 0:
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary[name] = (mean, std, len(vals))
    return summary


def paired_ttest(
    a: Dict[int, Dict[str, float]],
    b: Dict[int, Dict[str, float]],
) -> Dict[str, Tuple[float, float, int]]:
    """
    对两个配置/模型共有的指标做配对 t 检验。
    :return: {metric: (p_value, t_stat, n)}
    """
    from scipy import stats

    common_seeds = sorted(set(a.keys()) & set(b.keys()))
    if len(common_seeds) < 2:
        print("[warn] 两个配置的公共种子数 < 2，无法做配对 t 检验。")
        return {}

    a_metric_names = set()
    for s in common_seeds:
        a_metric_names.update(extract_metrics(a[s]).keys())
    b_metric_names = set()
    for s in common_seeds:
        b_metric_names.update(extract_metrics(b[s]).keys())

    out = {}
    for name in sorted(a_metric_names & b_metric_names):
        x = np.array([extract_metrics(a[s])[name] for s in common_seeds], dtype=float)
        y = np.array([extract_metrics(b[s])[name] for s in common_seeds], dtype=float)
        if len(x) < 2:
            continue
        t_stat, p_value = stats.ttest_rel(x, y)
        out[name] = (float(p_value), float(t_stat), len(x))
    return out


def print_summary(
    label: str,
    summary: Dict[str, Tuple[float, float, int]],
    metrics: Optional[List[str]] = None,
):
    print(f"\n===== {label} =====")
    print(f"{'metric':<20}{'mean':>10}{'std':>10}{'n':>5}")
    for name in metrics if metrics is not None else summary.keys():
        if name not in summary:
            continue
        mean, std, n = summary[name]
        print(f"{name:<20}{mean:>10.4f}{std:>10.4f}{n:>5}")


def main():
    ap = argparse.ArgumentParser(description="E-5 统计显著性计算")
    ap.add_argument("--task", choices=list(TASK_DIRS.keys()), default="linksign")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True, help="模型名，如 SignDyGFormer / DyGFormer")
    ap.add_argument(
        "--pattern",
        required=True,
        help="result_save_name 匹配子串，如 'RAS-E.RASE-E.BTE-E.CNAS-E'",
    )
    ap.add_argument(
        "--compare",
        default=None,
        help="（可选）同模型另一配置的匹配子串，做配对 t 检验",
    )
    ap.add_argument(
        "--compare-model",
        default=None,
        help="（可选）另一模型的名称，用相同 pattern 做配对 t 检验",
    )
    ap.add_argument("--prefix", default="test ", help="指标前缀，默认 'test '")
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="（可选）只输出指定指标，默认输出全部",
    )
    ap.add_argument("--output", default=None, help="（可选）输出 CSV 路径")
    args = ap.parse_args()

    base = os.path.join(".", "saved_results", TASK_DIRS[args.task], args.model, args.dataset)
    if not os.path.isdir(base):
        print(f"[error] 目录不存在: {base}")
        return

    results = load_seed_results(base, args.pattern)
    if not results:
        print(f"[error] 未找到匹配 pattern='{args.pattern}' 的结果文件: {base}")
        return
    print(f"找到 {len(results)} 个种子: {sorted(results.keys())}")

    summary = summarize(results)
    print_summary(f"{args.model} / {args.dataset} / {args.pattern}", summary, args.metrics)

    # 比较
    if args.compare is not None:
        results_b = load_seed_results(base, args.compare)
        if not results_b:
            print(f"[error] 未找到对比 pattern='{args.compare}' 的结果文件: {base}")
        else:
            print_summary(
                f"对比配置 {args.model} / {args.dataset} / {args.compare}",
                summarize(results_b),
                args.metrics,
            )
            pvals = paired_ttest(results, results_b)
            print(f"\n===== 配对 t 检验（{args.pattern} vs {args.compare}）=====")
            print(f"{'metric':<20}{'p_value':>10}{'t_stat':>10}{'n':>5}")
            for name in args.metrics if args.metrics is not None else pvals.keys():
                if name not in pvals:
                    continue
                p, t, n = pvals[name]
                print(f"{name:<20}{p:>10.4f}{t:>10.4f}{n:>5}")

    if args.compare_model is not None:
        base_b = os.path.join(
            ".", "saved_results", TASK_DIRS[args.task], args.compare_model, args.dataset
        )
        if not os.path.isdir(base_b):
            print(f"[error] 目录不存在: {base_b}")
        else:
            results_b = load_seed_results(base_b, args.pattern)
            if not results_b:
                print(f"[error] 未找到模型 {args.compare_model} pattern='{args.pattern}' 的结果文件")
            else:
                print_summary(
                    f"对比模型 {args.compare_model} / {args.dataset} / {args.pattern}",
                    summarize(results_b),
                    args.metrics,
                )
                pvals = paired_ttest(results, results_b)
                print(f"\n===== 配对 t 检验（{args.model} vs {args.compare_model}）=====")
                print(f"{'metric':<20}{'p_value':>10}{'t_stat':>10}{'n':>5}")
                for name in args.metrics if args.metrics is not None else pvals.keys():
                    if name not in pvals:
                        continue
                    p, t, n = pvals[name]
                    print(f"{name:<20}{p:>10.4f}{t:>10.4f}{n:>5}")

    # 可选 CSV 输出
    if args.output:
        import csv

        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "dataset", "config", "metric", "mean", "std", "n"])
            for cfg, label in [(args.pattern, args.model)]:
                for name, (mean, std, n) in summarize(results).items():
                    if args.metrics is not None and name not in args.metrics:
                        continue
                    writer.writerow([label, args.dataset, cfg, name, f"{mean:.4f}", f"{std:.4f}", n])
            if args.compare is not None:
                for name, (mean, std, n) in summarize(results_b).items():
                    if args.metrics is not None and name not in args.metrics:
                        continue
                    writer.writerow([args.model, args.dataset, args.compare, name, f"{mean:.4f}", f"{std:.4f}", n])
            if args.compare_model is not None:
                for name, (mean, std, n) in summarize(results_b).items():
                    if args.metrics is not None and name not in args.metrics:
                        continue
                    writer.writerow([args.compare_model, args.dataset, args.pattern, name, f"{mean:.4f}", f"{std:.4f}", n])
        print(f"\nCSV 已保存: {args.output}")


if __name__ == "__main__":
    main()
