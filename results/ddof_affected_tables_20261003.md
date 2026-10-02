# ddof 统一影响清单（Paper 2026-10-02 裁定：统一 ddof=1）

> 生成：Code 2026-10-03 · 对应裁定 `mb-20261002-173209-paper-d661 §四`
> 规则：这些工具原先用 **ddof=0（总体标准差）**，比主表/训练脚本的 **ddof=1（样本标准差）** 小 **√(4/5) = 0.8944**。
> ⇒ **受污染值换算：σ_正确 = σ_旧 / 0.8944 = σ_旧 × 1.1180**（n=5）。mean 不受影响。

## 一、已修改的工具（12 个，全部改为 ddof=1 / `st.stdev`）

| # | 文件 | 改动 | 主要输出 |
|---|---|---|---|
| 1 | `tools/verify/t13_cns_rows.py` | `.std(ddof=0)` → `ddof=1` | CNS 行值表（**已交付**：`caad`/`17bf`/`598d` 等） |
| 2 | `tools/verify/baseline_compare_table.py` | 同上 | 真基线/5 方法对照表（**已交付**：早期 S1 批） |
| 3 | `tools/verify/export_valthr_tables.py` | 同上 + 文案 "pstd(ddof=0)" → "std(ddof=1)" | val-thr 导出表（**已交付**：sign val-thr 批） |
| 4 | `tools/verify/e1a_pair_table.py` | 显示用 ± 改 ddof=1（配对 t 的 sd 本就是 ddof=1） | E1a 配对表（**已交付**：`19cb` 组合批） |
| 5 | `tools/verify/ras_radius_table.py` | 文案 ddof=0 → ddof=1 | 半径表（**已交付**：半径批） |
| 6 | `tools/verify/semba_variant_table.py` | `.std(ddof=0)` → `ddof=1` | SEMBA 变体表（**已交付**：SEMBA 批） |
| 7 | `tools/verify/t14b_full_keys.py` | 同上 | 全键核查表（**已交付**：D1/D2 附表） |
| 8 | `tools/verify/t17_metric_slice.py` | 同上 | 指标切片表（D1/D2 附表） |
| 9 | `tools/verify/thr_drift_table.py` | 同上（**Baseline 清单未含，我方自查追加**） | 阈值漂移表（内部核查） |
| 10 | `tools/verify/snapshot_results.py` | `st.pstdev` → `st.stdev`（**追加**） | 快照结果打印（内部） |
| 11 | `tools/verify/nh5_report.py` | 文案 "pstd(ddof=0) 与主表一致" → "std(ddof=1) 与主表一致"（**追加**，代码本就 ddof=0） | NH5 报告（内部） |
| 12 | `tools/verify/agg_configs.py` | 文案同上（**追加**） | 配置聚合表（内部） |

- 复核：`grep -n "ddof=0\|pstdev" tools/verify/*.py` ⇒ **无残留**（2026-10-03 00:12）。
- **未受影响**（本就 ddof=1）：`c1c2_table.py`、`cns_g2_table.py`、`wave2_mamba_table.py`、`wave2_scadyg_table.py`、`paired_stats_*.py`、`gconfirm_*`、`bte_margin_table.py`、`e1x_combo_verdict.py`、`grid_new_audit.py`、`_recon_stats.py`。
- **主表/C1-C2 表/训练脚本**本就 ddof=1 ✓（未被污染）。

## 二、对稿内 ± 的影响判定（供逐条核）

1. **凡由上表工具产出的 ± 都偏小 10.6%**；**mean、Δ、p、t、d 不受影响**（配对统计内部 sd 一直用 ddof=1）。
2. **换算**：`σ_正确 = 1.1180 × σ_旧`（n=5；若某表用的是 n≠5 的聚合，系数 = √(n/(n−1))）。
3. **已交付且含 ± 的产物（需你逐条核稿内引用）**：CNS 行值表、真基线/5 方法对照表、val-thr 导出表、E1a 配对表、半径表、SEMBA 变体表、D1/D2 附表（t14b/t17）。
4. **重生成方式**（纯后处理、秒级、无 GPU；等你确认后我一次性执行）：
   ```bash
   # 逐个重跑并覆盖受影响的 results/*.txt（工具均已修好）
   python tools/verify/t13_cns_rows.py > results/cns_rows_<date>.txt
   python tools/verify/baseline_compare_table.py > results/baseline_compare_<date>.txt
   python tools/verify/export_valthr_tables.py     # 自带落盘
   python tools/verify/e1a_pair_table.py ; python tools/verify/ras_radius_table.py
   python tools/verify/semba_variant_table.py ; python tools/verify/t14b_full_keys.py
   python tools/verify/t17_metric_slice.py
   ```
5. **建议**：稿内 ± 一律按 (2) 换算（等价于重跑），**或**等我重生成后整表替换 —— 两者结果一致（ddof 是确定性换算）。

## 三、结论（一句话给稿）

> 描述性 ± 的 ddof 口径已全仓统一为 **ddof=1**（与训练脚本/主表一致）。受影响的仅为**旧版 `tools/verify/*` 中 12 个汇总工具的显示用 σ**，其值偏小 10.6%（`σ_正确 = 1.1180 × σ_旧`）；**mean/Δ/t/p/d 不受影响**。
