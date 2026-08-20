"""
可插拔的随机符号翻转噪声模块（SignDyG 修订实验 E-7）
====================================================
设计目标：
- 可插拔：既可用于本仓库（DyGLib numpy 管线），也可整体迁移到 SEMBA 仓库（PyG TemporalData）。
- 公平性：flip_mask 基于【全局边顺序 + seed】确定性生成 → 两个仓库翻转完全一致的边集合。
- 编码感知：
  - SEMBA 侧：y ∈ {0,1}（预处理：1=正, 0=负）→ 翻转用 `1 - y`；
  - 本仓库侧：sign ∈ {-1,+1}（edge_feat 符号为原始幅值）→ 翻转用取负。

SEMBA 迁移用法（示例）：
    # 全数据加噪（InMemoryDataset 的 pre_transform 钩子）
    dataset = SEMBADataset(..., pre_transform=SignFlipNoise(0.1, seed=0))

    # train-only 加噪（在 train_val_test_split 之后，只对训练子集）
    data = dataset[0].to(device)
    train_data, val_data, test_data = data.train_val_test_split(
        val_ratio=0.15, test_ratio=0.15
    )
    SignFlipNoise(0.1, seed=0).apply_temporal(train_data)   # 只翻转 train 的 y

本仓库用法（示例）：
    python train_link_sign_prediction.py ... --noise-ratio 0.1 --noise-seed 0 --noise-scope train
    python train_sign_link_3class_prediction.py ... --noise-ratio 0.3 --noise-seed 0 --noise-scope all
"""

from enum import Enum

import numpy as np


class NoiseType(str, Enum):
    """噪声类型（SIGN_FLIP 已实现，其余预留扩展）"""

    SIGN_FLIP = "sign_flip"
    # SPURIOUS_EDGES = "spurious_edges"
    # MISSING_INTERACTIONS = "missing_interactions"


class NoiseScope(str, Enum):
    """噪声施加范围"""

    TRAIN = "train"  # 仅训练集加噪（val/test 保持干净）
    ALL = "all"      # 全数据加噪（train/val/test 全部）


class SignFlipNoise:
    """随机符号翻转噪声（可插拔：PyG TemporalData 与 numpy 双兼容）"""

    def __init__(self, noise_ratio: float, seed: int = 0):
        assert 0.0 <= noise_ratio <= 1.0, (
            f"noise_ratio 必须在 [0,1]，收到: {noise_ratio}"
        )
        self.noise_ratio = noise_ratio
        self.seed = seed

    def flip_mask(self, n_edges: int) -> np.ndarray:
        """
        确定性选择要翻转的全局边索引。
        基于【全局边顺序 + seed】生成，与具体框架无关 →
        两边仓库（同一边顺序）翻转的边完全一致（公平对比的前提）。
        """
        if n_edges <= 0 or self.noise_ratio == 0.0:
            return np.array([], dtype=np.int64)
        rng = np.random.RandomState(self.seed)
        n_flip = min(int(round(self.noise_ratio * n_edges)), n_edges)
        return rng.choice(n_edges, size=n_flip, replace=False)

    # ---------- SEMBA 侧（PyG TemporalData）----------
    def apply_temporal(self, data, inplace: bool = True):
        """
        翻转 data.y（SEMBA: y ∈ {0,1}，1=正, 0=负 → 1-y）。
        :param data: 含 .y 张量（torch.Tensor）的对象，如 PyG TemporalData。
        """
        mask = self.flip_mask(len(data.y))
        if len(mask) == 0:
            return data
        if inplace:
            data.y[mask] = 1 - data.y[mask]
            return data
        import copy

        new_data = copy.copy(data)
        new_data.y = new_data.y.clone()
        new_data.y[mask] = 1 - new_data.y[mask]
        return new_data

    # ---------- 本仓库侧（DyGLib numpy 管线）----------
    def apply_numpy(self, sign: np.ndarray) -> np.ndarray:
        """
        原地翻转 sign 数组（±1 → 取负），返回被翻转的索引。
        :param sign: numpy 数组（可写），如 Data.node_interact_sign。
        """
        mask = self.flip_mask(len(sign))
        if len(mask) > 0:
            sign[mask] *= -1
        return mask

    def __repr__(self):
        return f"SignFlipNoise(noise_ratio={self.noise_ratio}, seed={self.seed})"
