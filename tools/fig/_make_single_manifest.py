"""_make_single_manifest.py — 生成单面板交付 MANIFEST（含全量 sha256）。"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results" / "grid_figures_single_20261002"
CSV = ROOT / "figures" / "fig_grid_heatmap.csv"

L: list[str] = []
w = L.append
w("# 网格热力图「单面板紧凑件」交付 MANIFEST（Code → Paper · 2026-10-02）")
w("")
w("## 内容")
w("- **12 组** = 3 数据集（RedditTitle / RedditBody / WikiRfA）× 2 任务（linksign / sign）× 2 指标（主指标 / AUC）")
w("  - linksign 主指标 = `f1wt`；sign 主指标 = `f1mac`；另一组 = `auc`")
w("- **每组 2 档尺寸 × 2 格式（pdf/png）** ⇒ 48 文件：")
w("  - **无后缀 = 按最终尺寸设计**（面板宽 **168 pt**、注释字号 **6.8 pt**、刻度 5.5 pt、轴名 6.0 pt）")
w("    ⇒ 以 `0.32\\textwidth`(≈154 pt) 置入时接近 1:1，最终字号 ≈6.5–6.8 pt（可读）；")
w("  - **`_natural` = 自然尺寸件**（面板宽 **400 pt**、注释 7.5 pt）⇒ 若按自然尺寸(≈0.8\\textwidth)置入则用这档。")
w("")
w("## 与 `8b2f` 要求的对照 / 一处算术说明（需你确认口径）")
w("1. **已去红框**（不再标注\"当前点\"）+ 面板内不再出现 dataset/task 重复标题与图例（保留 NN/LF 轴名，标题仅简短 metric 名）。")
w("2. **单面板独立文件** ✓，命名 `fig_grid_{task}_{ds}_{metric}.{pdf,png}`。")
w("3. ⚠️ **算术说明**：`8b2f §一.3` 同时给出\"页面宽 ≈380–420 pt + 单元格字号 ≈7–8 pt\"与\"缩到 `0.32\\textwidth`(≈154 pt)\"。")
w("   这两条互斥：按 400 pt 设计再缩到 154 pt 时 7.5 pt 会变成 **≈2.9 pt**（与上次 16 in/18.9 pt 缩到 ≈2.5 pt 同理）。")
w("   故交**两档**：默认件按\"**最终尺寸**\"设计（168 pt / 6.8 pt ⇒ 缩到 0.32\\textwidth 后仍 ≈6.5 pt）；")
w("   `_natural` 件按你字面给的 400 pt/7.5 pt 设计（供自然尺寸置入）。**请按你的实际置入宽度选档**。")
w("4. 小件为给 4 位小数留宽度，**去掉了色标条**（每格数值已标注；如需色标可用 `_natural` 件或我另加窄色标）。")
w("")
w("## 代际证据（可复现）")
w(f"- 重跑 `tools/fig/gen_grid_heatmaps.py` ⇒ `figures/fig_grid_heatmap.csv` sha256 = "
  f"**{hashlib.sha256(CSV.read_bytes()).hexdigest().upper()}**（与 10-01 交付**逐位相同**）")
w("- 数据源：`results/grid_new/raw/{linksign,sign}/{ds}/SignDyGFormer_seed42.NN-*.LF-*.RAS-E.RASE-E.BTE-E.CNAS-E.P1.TE.json`"
  "（new generation 2026-09-25；**单种子 seed42**；每组 25 格全覆盖）")
w("")
w("## Bitcoin 覆盖（回答 `8b2f §二`）")
w("- **同代际（乃至任何代际）都没有 Bitcoin 网格**：`grid_new/raw` 仅 RT/RB/WV；全库检索无 Bitcoin 网格文件。")
w("- ⇒ 按你的分支②：**3 datasets**，每图一行 3 个；图注写\"同代际网格覆盖 3 个数据集；Bitcoin 两集未参与本轮网格审计\"。")
w("")
w("## 文件 sha256（前 16 位）")
for f in sorted(D.iterdir()):
    if f.is_file() and f.name != "MANIFEST.md":
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        w(f"- {h[:16]}  {f.name}  ({f.stat().st_size} B)")
(D / "MANIFEST.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L[:40]))
print(f"... [ok] {D.name}/MANIFEST.md （{len(list(D.iterdir()))-1} 个图件已登记）")
