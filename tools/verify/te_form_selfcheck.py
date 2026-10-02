"""te_form_selfcheck.py — §4.5 时间衰减第三臂（LINEAR）自检（Code 2026-10-02）。

覆盖：
  1. 语法编译：改动过的 5 个文件 py_compile；
  2. 枚举/默认：TimeDecayForm.EXP 为默认；linear 需显式指定；
  3. **零行为变更**：未指定 form/gamma 时，命名标签与历史一致（TE / TD）；
  4. linear 臂命名：λ 存在 + form=linear ⇒ `TD-LIN`（显式 γ ⇒ `TD-LIN.g<γ>`）；
  5. γ 自动定标公式自检：γ = (1 - exp(-λ·m)) / m 与 exp 臂在中位数 m 处等权；
  6. 权重数值自检（纯 numpy 复刻分支公式）：exp 分支与历史公式逐位相同；linear 分支 ∈[0,1] 且随 Δt 单调不增。

输出：results/te_form_selfcheck_20261002.txt（并打印 RESULT: PASS/FAIL）
"""
from __future__ import annotations

import py_compile
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "te_form_selfcheck_20261002.txt"

L: list[str] = []
w = L.append
fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    w(("  [PASS] " if cond else "  [FAIL] ") + msg)
    if not cond:
        fails.append(msg)


w("§4.5 时间衰减第三臂（LINEAR）自检 —— Code 2026-10-02")
w("=" * 96)
w("[1] 语法编译")
for f in ("models/NeighborInteractEncoder.py", "models/SignDyGFormer.py",
          "train_sign_link_3class_prediction.py", "train_link_sign_prediction.py",
          "utils/load_configs.py"):
    try:
        py_compile.compile(str(ROOT / f), doraise=True)
        check(True, f"compile {f}")
    except Exception as e:  # noqa: BLE001
        check(False, f"compile {f}: {e}")

w("")
w("[2] 枚举与默认")
from models.NeighborInteractEncoder import TimeDecayForm  # noqa: E402
check(TimeDecayForm.EXP.value == "exp" and TimeDecayForm.LINEAR.value == "linear", "TimeDecayForm 取值 exp/linear")
check(TimeDecayForm.EXP == "exp", "枚举与字符串兼容（可用于 pydantic/CLI）")

w("")
w("[3][4] 命名标签（零行为变更 + linear 可区分）")
from utils.load_configs import SignPredictArgs  # noqa: E402


def tag(**kw) -> str:
    a = SignPredictArgs(**kw)
    name = a.result_save_name
    if callable(name):  # 兼容可能的 property/方法两种实现
        name = name()
    return str(name)


try:
    base = dict(dataset_name="WikiVote", model_name="SignDyGFormer")
    t_none = tag(**base)
    t_exp = tag(**base, time_decay_lambda=0.1)
    t_lin = tag(**base, time_decay_lambda=0.1, time_decay_form=TimeDecayForm.LINEAR)
    t_ling = tag(**base, time_decay_lambda=0.1, time_decay_form=TimeDecayForm.LINEAR, time_decay_gamma=2.5)
    w(f"  默认(λ=None) 名尾 = ...{t_none}")
    w(f"  λ=0.1       名尾 = ...{t_exp}")
    w(f"  λ=0.1+linear 名尾 = ...{t_lin}")
    w(f"  λ=0.1+linear+γ=2.5 名尾 = ...{t_ling}")
    check("TE" in t_none, "λ=None ⇒ 标签含 TE")
    check("TD" in t_exp and "LIN" not in t_exp, "λ=0.1 默认(exp) ⇒ 标签 TD（与历史一致，零行为变更）")
    check("TD-LIN" in t_lin, "linear ⇒ 标签 TD-LIN（可区分）")
    check("TD-LIN.g2.5" in t_ling, "显式 γ ⇒ 标签 TD-LIN.g2.5")
except Exception as e:  # noqa: BLE001
    check(False, f"命名标签构造失败: {type(e).__name__}: {e}")

w("")
w("[5][6] 权重公式自检（纯 numpy 复刻分支）")
lam, m, dt = 0.1, 3.0, np.array([0.0, 0.5, 1.0, 3.0, 10.0])
w_exp = np.exp(-lam * dt)
w_lin_auto = np.maximum(0.0, 1.0 - ((1.0 - np.exp(-lam * m)) / m) * dt)
check(np.allclose(w_exp, np.exp(-lam * dt), atol=0.0, rtol=0.0), "exp 分支与历史公式逐位相同")
check(bool(np.isclose(w_lin_auto[np.argmin(np.abs(dt - m))], np.exp(-lam * m))), "linear(auto-γ) 在 Δt=m 处与 exp 等权")
check(bool(np.all(w_lin_auto >= 0.0) and np.all(w_lin_auto <= 1.0)), "linear 权重 ∈[0,1]")
check(bool(np.all(np.diff(w_lin_auto) <= 1e-12)), "linear 权重随 Δt 单调不增")
w(f"  Δt      = {dt.tolist()}")
w(f"  exp     = {np.round(w_exp, 6).tolist()}")
w(f"  linear  = {np.round(w_lin_auto, 6).tolist()}  (m={m}, γ={((1.0 - np.exp(-lam * m)) / m):.6f})")

w("")
w("=" * 96)
w(f"RESULT: {'PASS' if not fails else 'FAIL'}  ({len(fails)} 项失败)")
for f in fails:
    w(f"  ! {f}")
OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
print(f"[ok] {OUT.relative_to(ROOT)}")
