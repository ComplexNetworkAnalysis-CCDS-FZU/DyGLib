"""可选加速后端接入（M4 集成落地；设计：`D:\\codes\\SignDyG-Perf\\M4_INTEGRATION_DESIGN.md` §2）。

开关（命令行参数优先，其次环境变量；**默认启用**——用户 2026-09-12 口径）：
  - 命令行：训练脚本 `--accel / --no-accel`（经 `load_configs` 在参数解析后调用 `configure()`）；
  - 环境变量 `SIGNDYG_ACCEL`：`"1"`=启用 / `"0"`=关闭 / 未设=默认启用；
  - 启用 = **fail-fast**：未安装 / 契约不符 / 载入失败一律报错退出，**绝不静默回退**；
  - 关闭 = 100% 原始 Python 路径（连内核 import 都不尝试）。

契约串：`signdyg_accel.__abi__ == ABI`（内核语义变更必须同步 bump；不符 = 崩溃）。
实验溯源：结果 JSON 记录 `status`（{"enabled": bool, "abi": str|None}）。

实现说明：解析是**惰性**的（首次访问 on/kernel/status 或显式 `configure()` 时完成）——
这样命令行参数可以在程序启动早期（参数解析后）决定开关，随后于首次使用时严格校验；
已解析后不允许冲突重配（防中途切换语义）。
"""
import os

ABI = "k1k2-v2-2026-09-11"
ENV_VAR = "SIGNDYG_ACCEL"

_resolved = False


def _env_default():
    raw = os.environ.get(ENV_VAR, "")
    if raw == "0":
        return False
    if raw in ("", "1"):
        return True
    raise RuntimeError(f"[accel] 非法 {ENV_VAR}={raw!r}（仅支持 '0'/'1'）")


def configure(enabled=None):
    """配置加速开关。enabled: True/False 显式指定；None = 回落环境变量（默认启用）。

    进程内首次生效；重复配置仅允许同值（冲突时报错，防止运行中途切换）。
    """
    global _resolved, kernel, on, status
    want = _env_default() if enabled is None else bool(enabled)
    if _resolved:
        if want != on:
            raise RuntimeError("[accel] 开关已确定后不允许冲突重配")
        return
    if want:
        import signdyg_accel  # 未安装 → ModuleNotFoundError（fail-fast：拒绝运行）
        abi = getattr(signdyg_accel, "__abi__", None)
        if abi != ABI:
            raise RuntimeError(
                f"[accel] 契约不符：__abi__={abi!r}，期望 {ABI!r}——请重建/更新 signdyg_accel"
            )
        kernel = signdyg_accel
        on = True
        print(f"[accel] enabled（signdyg_accel __abi__={abi}）")
    else:
        kernel = None
        on = False
        print("[accel] disabled（原始路径）")
    status = {"enabled": on, "abi": (ABI if on else None)}
    _resolved = True


def __getattr__(name):  # 惰性解析：on / kernel / status 在首次访问前不存在
    if name in ("on", "kernel", "status"):
        configure(None)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
