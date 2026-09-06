# GPU 驱动修复操作文件（2026-09-06）

> ✅ **执行结果：已完成（2026-09-06）** —— 实际安装 **DKMS `nvidia/595.84`**（非计划中的 590），
> `torch.cuda.is_available()=True`，设备 `NVIDIA GeForce RTX 2080 SUPER`，fedsa 免 sudo 可用。
> 本文件按"审阅后执行"编写；实际安装路径/版本与下方计划略有出入，以本行结论为准。

> 服务器：`fedsa@172.17.173.102`（Ubuntu 22.04.3 LTS / HWE 内核 `6.8.0-124-generic` / 物理机 / UEFI / SecureBoot **disabled**）
> GPU：2× RTX 2080 SUPER（TU104）
> 授权：服务器管理同学已授权安装显卡驱动（2026-09-06）
> 目标：**免重启** 加载 nvidia 内核模块，使 `nvidia-smi` / CUDA 可用，不中断正在运行的 E-3（CPU 训练）。
>
> ⚠️ **本文件仅供审阅。用户逐条确认后，方可执行。**

## 根因（诊断已确认）

| 事实 | 状态 |
|---|---|
| 用户态驱动 590.48.01（nvidia-smi、libnvidia-*） | ✅ 已安装 |
| 现役内核 6.8.0-124 的 nvidia 内核模块 | ❌ 缺失（`modinfo nvidia` → not found） |
| 已装包 `linux-modules-nvidia-590-6.8.0-110` | ⚠️ 空包（无 `.ko`，110 内核也无模块） |
| apt 源 `nvidia-dkms-590`（590.48.01-0ubuntu0.22.04.4） | ✅ 可安装（DKMS 源码包，按现役内核编译） |
| 现役内核头文件 `linux-headers-6.8.0-124-generic` / gcc 11.4 / make 4.3 | ✅ 齐全 |
| SecureBoot | ✅ disabled（无模块签名障碍） |
| E-3 训练进程 | 🔄 运行中（CPU）→ **禁止重启、禁止误杀** |

**结论**：采用 **DKMS 免重启** 路线 —— 安装 `nvidia-dkms-590`，为**现役内核**编译模块后 `modprobe` 加载。全程不需要重启。

---

## 风险等级图例

- 🟢 **低**：可逆、影响面小、失败不影响现有系统
- 🟡 **中**：有状态变化，但可回滚；失败有明确报错与恢复路径
- 🔴 **高**：可能中断服务/需重启/难回滚，执行前必须二次确认

---

## Phase 0 — 前置快照与确认（🟢 全低，只读）

| # | 操作 | 命令 | 风险 | 说明 |
|---|---|---|---|---|
| 0.1 | 快照当前包/模块状态（留档，便于回滚比对） | `dpkg -l > /tmp/dpkg_before.txt; lsmod > /tmp/lsmod_before.txt; modinfo nvidia 2>&1 > /tmp/modinfo_before.txt` | 🟢 | 纯只读留档 |
| 0.2 | 记录 E-3 训练 PID（防误伤） | `pgrep -af 'train_sign_link_3class'` | 🟢 | 记住 PID，后续任何操作不 kill |
| 0.3 | 与管理员确认边界 | 手动沟通 | 🟢 | 确认：允许 apt 装包、允许加载内核模块、**允许从 apt 装 nvidia-dkms-590**；本方案不请求重启 |

---

## Phase 1 — 刷新 apt 缓存（🟢 低）

| # | 操作 | 命令 | 风险 | 说明 |
|---|---|---|---|---|
| 1.1 | 刷新软件源列表 | `sudo apt update` | 🟢 | 仅刷新 `/var/lib/apt/lists`；不影响运行中进程。失败：网络/repo 报错 → 检查网络与源配置，重试即可 |

---

## Phase 2 — 安装 nvidia-dkms-590（🟡 中，核心步骤，免重启）

| # | 操作 | 命令 | 风险 | 说明 |
|---|---|---|---|---|
| 2.1 | 安装 DKMS 驱动源码包 | `sudo apt install -y nvidia-dkms-590` | 🟡 | 下载 + 为现役内核 6.8.0-124 **自动编译**（5–15 分钟）。连带把用户态 590 从 .3 升到 .4（同一大版本，安全）。**风险①**：编译失败（内核 API 不兼容）→ 见 2.2 验证与回滚；**风险②**：apt 提示依赖冲突（少见）→ 先 `apt-get -f install` 修复再重试 |
| 2.2 | 验证编译产物 | `dkms status` | 🟢 | 期望见 `nvidia/590.48.01, 6.8.0-124-generic, ... installed`。若为 `failed`/`built` 未 install → 查 `/var/lib/dkms/nvidia/*/build/make.log` 末尾；**回滚**：`sudo apt remove -y nvidia-dkms-590`（仅删内核模块，用户态 590 保留，系统回到现状） |

> 🟡 **风险提示（2.1 关键）**：dkms 编译是唯一可能耗时/失败的环节。590 驱动较新，对 6.8 内核兼容性好，预计顺利；即便失败，回滚干净（`apt remove` 即还原），**不会破坏现有系统与 E-3**。

---

## Phase 3 — 加载内核模块（🟡 中）

| # | 操作 | 命令 | 风险 | 说明 |
|---|---|---|---|---|
| 3.1 | 加载 nvidia 主模块 | `sudo modprobe nvidia` | 🟡 | 失败（如版本/内核不匹配）→ 报错明确，回到 Phase 2 查 dkms 日志。加载本身**可逆**：`sudo modprobe -r nvidia` |
| 3.2 | 加载配套模块 | `sudo modprobe nvidia_uvm nvidia_modeset nvidia_drm` | 🟡 | uvm 为 CUDA 必需；modeset/drm 供显示（本机 headless 可不强求 drm）。失败同样可 `-r` 卸载 |
| 3.3 | 验证 GPU 可见 | `nvidia-smi` | 🟢 | 期望显示 2× RTX 2080 SUPER + 驱动版本 590.48.01。若仍 "NVML: Driver not loaded" → 模块未生效，查 `dmesg \| tail` |
| 3.4 | 确认开机自加载 | `systemctl list-unit-files \| grep -i nvidia` 或重启后验证（本方案不重启，先靠本次加载生效；nvidia-dkms 安装通常会配置开机加载） | 🟢 | 记录现状即可，下次重启后另验 |

---

## Phase 4 — 普通用户访问 GPU（🟢 低）

| # | 操作 | 命令 | 风险 | 说明 |
|---|---|---|---|---|
| 4.1 | 确保 fedsa 可访问设备 | `sudo usermod -aG video fedsa`（若尚未在 video 组） | 🟢 | 设备节点 udev 规则通常授权 video 组；加组立即生效于新会话。回滚：`sudo deluser fedsa video` |
| 4.2 | 免 sudo 验证 | `nvidia-smi`（重新登录后） | 🟢 | 无需 sudo 能显示 = 权限 OK |

---

## Phase 5 — torch/CUDA 验证（🟢 低，不抢 E-3 资源）

| # | 操作 | 命令 | 风险 | 说明 |
|---|---|---|---|---|
| 5.1 | gc 环境 CUDA 可用性 | `source ~/.bashrc; conda activate gc; python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"` | 🟢 | 期望 `True / NVIDIA GeForce RTX 2080 SUPER`。首次 CUDA 调用会初始化，耗时数秒正常 |
| 5.2 | 确认 GPU 空闲再调度 | `nvidia-smi` | 🟢 | E-3 在 CPU 跑，GPU 应全空闲；确认无其他租户任务后再启动 GPU 实验 |
| 5.3 | （建议）小规模冒烟 | 单 seed、小数据集短跑一次 | 🟢 | 验证训练管线在 GPU 下正常，再决定是否切换/重排后续实验 |

---

## Phase 6 — 备用回退路径（🔴 高，仅当 Phase 2/3 失败且需重启时）

> **前提**：仅当 DKMS 编译彻底失败且 `apt remove` 回滚后仍无法解决，才考虑以下路径；**执行前必须与管理员确认 + 用户二次批准**，因为会中断 E-3。

| # | 操作 | 风险 | 说明 |
|---|---|---|---|
| 6.1 | 尝试 apt 安装预编译模块包 `linux-modules-nvidia-590-6.8.0-124-generic`（若源内有） | 🔴 | 现 `apt-cache policy` 显示该包**不可用**（源内没有 124 的预编译模块），大概率走不通；且旧 110 包已被证实是空包 |
| 6.2 | 从 NVIDIA 官网 `.run` 安装 | 🔴 | 需先卸载现有 590 用户态包（`apt purge nvidia-*`），改动面大、易留残留；**不推荐** |
| 6.3 | 重启到有模块的旧内核 | 🔴 | 现役 124 与旧 110 均无现成模块（110 包为空），重启收益低且**中断 E-3**；**不推荐** |

---

## 风险总览

| 风险点 | 等级 | 缓解 |
|---|---|---|
| dkms 编译失败 | 🟡 | 回滚 = `apt remove nvidia-dkms-590`，系统还原，E-3 不受影响 |
| apt 依赖冲突 | 🟡 | `apt-get -f install` 修复；再不行回滚 |
| 加载模块与内核不匹配 | 🟡 | `modprobe -r` 卸载还原；查 dmesg |
| 误伤 E-3 | 🔴（预防） | 全程不 kill、不重启；0.2 先记 PID |
| 需重启才能生效 | 🔴（备用） | 本方案设计免重启，重启仅作 Phase 6 最后手段且须二次批准 |
| 与管理员授权范围不符 | 🟡 | 0.3 先口头确认装包范围 |

---

## 执行前置条件清单

- [ ] 用户逐条审阅并批准本文件
- [ ] 0.3 管理员确认装包范围
- [ ] 记录 E-3 PID（0.2）
- [ ] sudo 密码由**用户本人**在终端输入（不经 AI/聊天工具传递）

## 执行方式建议

sudo 命令需密码，**密码不经 AI 传递**。两种执行方式任选：
1. **用户亲跑**：按 Phase 顺序在服务器终端逐条执行（本文件每步命令即抄即用），AI 在旁指导验证；
2. **脚本化**：把 Phase 1–5 整理成 `gpu_driver_fix.sh`，用户 `sudo bash` 执行；AI 只读验证输出。

> 注：执行后如成功，建议顺手把 Phase 0.1 快照与结果写入 `docs/PROGRESS.md`（服务器环境节），并通知 Agent A 耗时预期可改为 GPU 口径。
