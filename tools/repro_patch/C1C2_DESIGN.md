# C1/C2 可见性对照实现设计（Code 草案 · 2026-09-30 夜）

> 状态：**草案**（未落库）。开工条件：Baseline 无异议（异议截止 10-01 09:00）→ 在**服务器 repro 工作副本分支**实现 + patch 交付。
> 口径权威：`docs/ADVISOR_DECISIONS.md` + 信箱 `0915`。禁止动 `main`。

## 0. 已核代码事实（只读复核，`train_eval/dysdgnn.py` + `models/dysdgnn/model.py`）

| 事实 | 位置 | 含义 |
|---|---|---|
| `active_clips[node][:searchsorted(al, k, side="right")]` | `dysdgnn.py::_prep` | MT-SA 序列**含当前 clip k 自身** |
| `nbr_lists` 由**本 clip k 的全部边**构建 | `dysdgnn.py::_prep` | 目标边自身作为 motif 邻居可见（自泄漏） |
| `encode_clips` **逐 clip 顺序递推**，`layer_buf[l]` 携带 k−1 状态 | `model.py:132` | z_k = f(z_{<k} 链, clip k 自身 motif 输入, dt_k) ⇒ **可增量：前 k−1 步状态与"是否掩蔽 clip k"无关** |
| `_eval_split` = `mtsa_at(z_stack, clips, k)` 后直接打分 clip k 的边 | `dysdgnn.py::_eval_split` | **C0 = 转导锚**（目标边参与自身嵌入） |

## 1. 三档定义（交付口径）

| 档 | 语义 | 精确实现 | `approx` |
|---|---|---|---|
| **C0** | 转导锚：现状（目标边参与自身嵌入） | 现有代码路径，位级复现既有数 | `false` |
| **C1** | **逐边精确 mask-self**：仅剔除该目标边，窗口内其余边仍可见 | 对每条测试边 e，在 clip k 的 motif 邻居表中剔除 e（两端点各删 1 条），用**未掩蔽链**到 k−1 的状态 + 掩蔽后的 clip k 前向一次，再打分 e | `false` |
| **C2** | **严格过去**：clip k 全部信息不可见（邻居只取自 clips<k） | 逐窗一次前向：维持在 k−1 的**未掩蔽链**缓冲，对 clip k 用"过去邻居表"前向一次，给该窗全部边打分 | `false` |
| ~~C1-approx / C2-approx~~ | 全局剔测试边 / 前窗为界 | ⚠️ 在本设计下**退化为同一构造**（等宽窗 + 整窗测试集）→ 只作鲁棒性，须 `approx=true` + `approx_kind` | `true` |

## 2. 递推实现要点（关键正确性论证）

`encode_clips` 的 `layer_buf` 在 clip k 处只依赖 `clips[0..k-1]` 的**未掩蔽**结果。因此：

- **C2 精确 = 一趟增量**：按窗顺序推进"未掩蔽链"到 k−1（缓存），再对 clip k 做一次"过去邻居"前向 → 每窗恰一次前向，O(Σ|active|)。
- **C1 精确 = 每测试边一次前向**：同一条未掩蔽链 + 掩蔽 clip k 的两端点邻居 → 每边一次（估 10–50 ms × 3–5k 边 ≈ +2–5 min/run）。
- 掩蔽不得污染链：掩蔽前向**不写回** `layer_buf`（保持未掩蔽链），否则 k+1 会继承被掩蔽状态（C2 会失去"每窗独立"性质）。

## 3. 口令与产物契约

```
--eval-protocol {C0,C1,C2}          # 默认 C0（不传 = 旧行为，位级一致）
--eval-approx {none,global-mask,past-window}   # 非 none 时 approx=true
--eval-protocol-out <json>          # 或并入既有结果 JSON
```
每 run JSON 必含：
```json
{"eval_protocol": "C1", "approx": false, "approx_kind": null,
 "n_eval_edges": 4123, "protocol_detail": "per-edge mask-self; unmasked past chain"}
```

## 4. 自检（开工后先跑，再全量）

1. **C0 复现**：改后以 `--eval-protocol C0` 跑 RB 单 seed，须与既有 `results/` 数值**位级一致**（回归闸门）。
2. **C2 链不污染**：断言"掩蔽窗不影响下一窗的未掩蔽状态"（对同 seed 比较 k+1 值，与只跑到 k 的独立复算一致）。
3. **C1 单调性自检**：对同一目标边，C1 分数 ≥ C2 分数不应恒成立（仅作异常检测）；抽样 20 条边逐边手算核对。
4. **规模**：3 ds × 5 seed × 3 档 = 45 runs（优先 RB 与 sign RT，支撑附表 DySDGNN F1_bin 退化解释）。

## 5. 交付物

- 分支 `eval-protocol-c1c2`（禁止 main）；`git diff main..分支 > tools/repro_patch/c1c2_<日期>.patch`；
- `results/` 产物 + sha256 入账（`results/_sync_raw_log.csv`）；
- 结论：**哪些结论跨档改变**（预期 C1≈C0、C2 与 C0 有差 → 说明可见性影响量级）。
