from enum import Enum
from typing import Callable, Optional, Union
from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from utils import accel as _accel


class EncodeType(Enum):
    CoOccurredNeighbor = 0
    InteractSignEffect = 1


class TimeDecayGapMode(str, Enum):
    """E-4 时间衰减 Δt 的定义方式（使用枚举避免 typo 导致实验错误）"""

    # A: 证据陈旧度 t_query - max(t_uk, t_kv)（较新事件相对查询时刻的间隔）
    STALENESS = "staleness"
    # B: 两条历史事件的时间间隔 |t_uk - t_kv|
    GAP = "gap"


class NeighborCooccurrenceEncoder(nn.Module):

    def __init__(
        self,
        neighbor_co_occurrence_feat_dim: int,
        device: str = "cpu",
        *,
        module_repeat_aware_sign_encoder: bool = False,
        module_balance_theory_encoder: bool = True,
        module_common_neighbor_encoder: bool = True,
        time_decay_lambda: Optional[float] = None,
        time_decay_gap_mode: TimeDecayGapMode = TimeDecayGapMode.STALENESS,
        time_scaling_factor: float = 1e-6,
    ):
        """
        Neighbor co-occurrence encoder.
        :param neighbor_co_occurrence_feat_dim: int, dimension of neighbor co-occurrence features (encodings)
        :param device: str, device
        :param module_common_neighbor_encoder: bool, 共同邻居编码（CNE，共现特征）通道开关；
            False = 通道特征置零（探针，切信息不切结构）；默认 True = 零行为变更
        :param time_decay_lambda: Optional[float], 时间衰减系数 λ；None 表示不启用时间衰减 (E-4)
        :param time_decay_gap_mode: TimeDecayGapMode, Δt 定义: STALENESS=A(证据陈旧度), GAP=B(事件间隔)
        :param time_scaling_factor: float, 时间归一化因子（与采样器一致）
        """
        super(NeighborCooccurrenceEncoder, self).__init__()
        self.neighbor_co_occurrence_feat_dim = neighbor_co_occurrence_feat_dim
        self.device = device
        self.module_repeat_aware_sign_encoder = module_repeat_aware_sign_encoder
        self.module_balance_theory_encoder = module_balance_theory_encoder
        # CNE 通道开关（2026-09-17 用户批准；Paper 6c7a 四）：False = 共现特征置零探针
        self.module_common_neighbor_encoder = module_common_neighbor_encoder
        # 时间衰减：lambda 为 None 即不启用（用 Optional 判断）
        self.time_decay_mode = time_decay_lambda is not None
        self.time_decay_lambda = time_decay_lambda
        # 仅接受枚举项，避免 typo 静默产生错误实验
        assert isinstance(time_decay_gap_mode, TimeDecayGapMode), (
            f"time_decay_gap_mode 必须是 TimeDecayGapMode 枚举项，收到: {time_decay_gap_mode!r}"
        )
        self.time_decay_gap_mode = time_decay_gap_mode
        self.time_scaling_factor = time_scaling_factor

        self.neighbor_co_occurrence_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.neighbor_co_occurrence_feat_dim),
            nn.ReLU(),
            nn.Linear(
                in_features=self.neighbor_co_occurrence_feat_dim,
                out_features=self.neighbor_co_occurrence_feat_dim,
            ),
        )

        self.neighbor_sign_effect_layer = nn.Sequential(
            nn.Linear(in_features=2, out_features=self.neighbor_co_occurrence_feat_dim),
            nn.LeakyReLU(negative_slope=0.5),
            nn.Linear(
                in_features=self.neighbor_co_occurrence_feat_dim,
                out_features=self.neighbor_co_occurrence_feat_dim,
            ),
        )

        with torch.no_grad():
            self.neighbor_sign_effect_layer[0].weight *= 10.0
            self.neighbor_sign_effect_layer[0].bias *= 10.0

    def sign_neighbor_count(self, node_neighbor_ids, node_neighbor_sign):
        all_ids, inverse_indexes = np.unique(node_neighbor_ids, return_inverse=True)

        pos_sample = node_neighbor_ids[node_neighbor_sign == 1]
        neg_sample = node_neighbor_ids[node_neighbor_sign == -1]

        pos_id, pos_count = np.unique(pos_sample, return_counts=True)
        neg_id, neg_count = np.unique(neg_sample, return_counts=True)

        pos_map = dict(zip(pos_id, pos_count))
        neg_map = dict(zip(neg_id, neg_count))

        pos_vec = np.array([pos_map.get(i, 0) for i in all_ids])
        neg_vec = np.array([neg_map.get(i, 0) for i in all_ids])

        pos_node_neighbor_counts = pos_vec[inverse_indexes]
        neg_node_neighbor_counts = neg_vec[inverse_indexes]
        signed_mapping_dict = dict(zip(all_ids, zip(pos_vec, neg_vec)))

        return pos_node_neighbor_counts, neg_node_neighbor_counts, signed_mapping_dict

    def sign_effect_count(
        self,
        *,
        common_neighbor: np.ndarray,
        src_node_neighbor_ids: np.ndarray,
        src_node_neighbor_sign: np.ndarray,
        dst_node_neighbor_sign: np.ndarray,
        dst_node_neighbor_ids: np.ndarray,
        src_node_neighbor_times: Optional[np.ndarray] = None,
        dst_node_neighbor_times: Optional[np.ndarray] = None,
        query_time: Optional[float] = None,
    ):
        """
        计算src-dst 之间的共同邻居对符号的影响
        - 使用平衡理论, 对于共同邻居 k, 节点对u,v 有以下情况
          - u + k  k + v = 正边影响 +1
          - u - k  k - v = 正边影响 +1
          - u - k  k + v = 负边影响 +1
          - u + k  k - v = 负边影响 +1
        - 时间衰减模式（time_decay_mode）下，每条三元组证据的贡献由 exp(-λ·Δt) 加权：
          - gap_mode='staleness' (A): Δt = t_query - max(t_uk, t_kv)，证据陈旧度
          - gap_mode='gap' (B):      Δt = |t_uk - t_kv|，两条历史事件的时间间隔

        :param common_neighbor: src dst 节点之间的共同邻居列表
        :type common_neighbor: np.ndarray
        :param src_node_neighbor_ids: 源节点的历史邻居列表
        :type src_node_neighbor_ids: np.ndarray
        :param src_node_neighbor_sign: 源节点的历史邻居交互符号
        :type src_node_neighbor_sign: np.ndarray
        :param dst_node_neighbor_sign: 目标节点的历史邻居交互符号
        :type dst_node_neighbor_sign: np.ndarray
        :param dst_node_neighbor_ids: 目标节点的历史邻居交互符号
        :type dst_node_neighbor_ids: np.ndarray
        """

        src_common_idx = np.isin(src_node_neighbor_ids, common_neighbor)  # 位置索引
        dst_common_idx = np.isin(dst_node_neighbor_ids, common_neighbor)

        src_common_neighbor_ids = src_node_neighbor_ids[src_common_idx]
        src_common_neighbor_sign = src_node_neighbor_sign[src_common_idx]
        src_common_neighbor_times = (
            src_node_neighbor_times[src_common_idx]
            if src_node_neighbor_times is not None
            else None
        )

        dst_common_neighbor_ids = dst_node_neighbor_ids[dst_common_idx]
        dst_common_neighbor_sign = dst_node_neighbor_sign[dst_common_idx]
        dst_common_neighbor_times = (
            dst_node_neighbor_times[dst_common_idx]
            if dst_node_neighbor_times is not None
            else None
        )

        src_idx = np.arange(len(src_common_neighbor_ids))
        dst_idx = np.arange(len(dst_common_neighbor_ids))

        # 展开
        src_idx_flat = src_idx.repeat(len(dst_common_neighbor_ids))
        dst_idx_flat = np.tile(dst_idx, len(src_common_neighbor_ids))

        src_sign_flat, dst_sign_flat = (
            src_common_neighbor_sign[src_idx_flat],
            dst_common_neighbor_sign[dst_idx_flat],
        )

        mask = (
            src_common_neighbor_ids[src_idx_flat]
            == dst_common_neighbor_ids[dst_idx_flat]
        )

        src_idx = src_idx_flat[mask]
        dst_idx = dst_idx_flat[mask]

        suggest_sign = src_sign_flat[mask] * dst_sign_flat[mask]

        # 去除不同共邻居后的正负
        pos_suggest_mask = suggest_sign == 1
        neg_suggest_mask = suggest_sign == -1

        # 时间衰减权重（可选）：每条三元组证据 × exp(-λ·Δt)
        triad_weights = None
        if (
            self.time_decay_mode
            and src_common_neighbor_times is not None
            and dst_common_neighbor_times is not None
            and query_time is not None
        ):
            src_time_flat = src_common_neighbor_times[src_idx_flat]
            dst_time_flat = dst_common_neighbor_times[dst_idx_flat]
            if self.time_decay_gap_mode == TimeDecayGapMode.GAP:
                # B: 两条历史事件的时间间隔
                dt = np.abs(src_time_flat - dst_time_flat)
            else:
                # A (default): 证据陈旧度（较新事件相对查询时刻的间隔）
                dt = query_time - np.maximum(src_time_flat, dst_time_flat)
                dt = np.maximum(dt, 0.0)
            dt = dt[mask]
            dt_scaled = dt * self.time_scaling_factor
            triad_weights = np.exp(-self.time_decay_lambda * dt_scaled)

        #
        pos_effect = src_common_neighbor_ids[src_idx_flat[mask][pos_suggest_mask]]
        neg_effect = src_common_neighbor_ids[src_idx_flat[mask][neg_suggest_mask]]

        if triad_weights is None:
            pos_keys, pos_count = np.unique(pos_effect, return_counts=True)
            neg_keys, neg_count = np.unique(neg_effect, return_counts=True)
            pos_effect = dict(zip(pos_keys, pos_count))
            neg_effect = dict(zip(neg_keys, neg_count))
        else:
            # 加权聚合：加权和替代整数计数
            pos_keys, pos_inverse = np.unique(pos_effect, return_inverse=True)
            neg_keys, neg_inverse = np.unique(neg_effect, return_inverse=True)
            pos_effect = dict(
                zip(
                    pos_keys,
                    np.bincount(
                        pos_inverse, weights=triad_weights[pos_suggest_mask]
                    ),
                )
            )
            neg_effect = dict(
                zip(
                    neg_keys,
                    np.bincount(
                        neg_inverse, weights=triad_weights[neg_suggest_mask]
                    ),
                )
            )

        return pos_effect, neg_effect

    def to_float_torch(self, arr: np.ndarray, apply: Union[Callable, None] = None):
        tensor = torch.from_numpy(arr)

        if apply is not None:
            tensor = tensor.apply_(apply)

        tensor = tensor.float().to(self.device)

        return tensor

    def count_neighbor_sign_effect(
        self,
        *,
        src_nodes: np.ndarray,
        dst_nodes: np.ndarray,
        src_padded_nodes_neighbor_ids: np.ndarray,
        dst_padded_nodes_neighbor_ids: np.ndarray,
        src_padded_nodes_neighbor_sign: np.ndarray,
        dst_padded_nodes_neighbor_sign: np.ndarray,
        node_interact_times: Optional[np.ndarray] = None,
        src_padded_nodes_neighbor_times: Optional[np.ndarray] = None,
        dst_padded_nodes_neighbor_times: Optional[np.ndarray] = None,
    ):
        # ---- M4 加速接缝（默认启用；--no-accel / SIGNDYG_ACCEL=0 关闭）----
        # 与下方原路径逐位一致（K2 bit-exact 门禁验证）；内核返回 f32[B,L,2] numpy。
        if _accel.on:
            src_effect, dst_effect = _accel.kernel.bte_sign_effect(
                src_nodes=src_nodes,
                dst_nodes=dst_nodes,
                src_padded_ids=src_padded_nodes_neighbor_ids,
                dst_padded_ids=dst_padded_nodes_neighbor_ids,
                src_padded_signs=src_padded_nodes_neighbor_sign,
                dst_padded_signs=dst_padded_nodes_neighbor_sign,
                src_padded_times=src_padded_nodes_neighbor_times,
                dst_padded_times=dst_padded_nodes_neighbor_times,
                query_times=node_interact_times,
                time_decay_lambda=self.time_decay_lambda,
                time_decay_gap_mode=self.time_decay_gap_mode.value,
                time_scaling_factor=self.time_scaling_factor,
                module_repeat_aware_sign_encoder=self.module_repeat_aware_sign_encoder,
                zero_padding=True,
            )
            return (
                torch.from_numpy(src_effect).to(self.device),
                torch.from_numpy(dst_effect).to(self.device),
            )
        src_padded_nodes_sign_effect, dst_padded_nodes_sign_effect = [], []
        # 对每个节点对（单个批次的每个节点）
        # src_padded_node_neighbor_ids, ndarray, shape (src_max_seq_length, )
        # dst_padded_node_neighbor_ids, ndarray, shape (dst_max_seq_length, )
        for (
            node_idx,
            (
                src_id,
                dst_id,
                src_padded_node_neighbor_ids,
                dst_padded_node_neighbor_ids,
                src_padded_node_neighbor_sign,
                dst_padded_node_neighbor_sign,
            ),
        ) in enumerate(
            zip(
                src_nodes,
                dst_nodes,
                src_padded_nodes_neighbor_ids,
                dst_padded_nodes_neighbor_ids,
                src_padded_nodes_neighbor_sign,
                dst_padded_nodes_neighbor_sign,
            )
        ):
            # ---- 修复 2026-09-09：封 pos0 标签泄漏 + indirect 只收真第三方 ----
            # pos0 = 自身 token，其 sign 原为当前待预测边标签；此前经共同邻居配对(suggest=标签×历史)
            # 造成测试时标签泄漏（重复边上指标虚高）。现在只用手历史切片 [1:] 参与交集与计数：
            # u/v 各自只出现在自己侧 pos0，切片后不再互相配对 → 交集自动只含真第三方 w∉{u,v}，
            # 与论文 §3.3.2 “counterpart 永不是共同邻居”的前提一致。
            src_hist_ids = src_padded_node_neighbor_ids[1:]
            dst_hist_ids = dst_padded_node_neighbor_ids[1:]
            src_hist_sign = src_padded_node_neighbor_sign[1:]
            dst_hist_sign = dst_padded_node_neighbor_sign[1:]
            src_hist_times = (
                src_padded_nodes_neighbor_times[node_idx][1:]
                if src_padded_nodes_neighbor_times is not None
                else None
            )
            dst_hist_times = (
                dst_padded_nodes_neighbor_times[node_idx][1:]
                if dst_padded_nodes_neighbor_times is not None
                else None
            )

            src_unique_keys = np.unique(src_hist_ids)
            dst_unique_keys = np.unique(dst_hist_ids)
            common_neighbor = np.intersect1d(src_unique_keys, dst_unique_keys)

            pos_effect, neg_effect = self.sign_effect_count(
                common_neighbor=common_neighbor,
                src_node_neighbor_ids=src_hist_ids,
                src_node_neighbor_sign=src_hist_sign,
                dst_node_neighbor_ids=dst_hist_ids,
                dst_node_neighbor_sign=dst_hist_sign,
                src_node_neighbor_times=src_hist_times,
                dst_node_neighbor_times=dst_hist_times,
                query_time=(
                    node_interact_times[node_idx]
                    if node_interact_times is not None
                    else None
                ),
            )
            # TODO: 重复信息：
            # 对于节点对 u， v , 如果历史上出现了重复的u,v
            # 那u这里，编码v的正负连边影响，v同理
            # 理论上应该是一样的,但是可能不一样
            # src_in_dst_sign = dst_padded_node_neighbor_sign[
            #     dst_padded_node_neighbor_ids == src_id
            # ]
            # dst_in_src_sign = src_padded_node_neighbor_sign[
            #     src_padded_node_neighbor_ids == dst_id
            # ]

            # TODO: 计算邻居符号方向影响

            src_neighbor_pos_effect = self.to_float_torch(
                src_padded_node_neighbor_ids.copy(),
                lambda neighbor_id: pos_effect.get(neighbor_id, 0.0),
            )
            src_neighbor_neg_effect = self.to_float_torch(
                src_padded_node_neighbor_ids.copy(),
                lambda neighbor_id: neg_effect.get(neighbor_id, 0.0),
            )

            dst_neighbor_pos_effect = self.to_float_torch(
                dst_padded_node_neighbor_ids.copy(),
                lambda neighbor_id: pos_effect.get(neighbor_id, 0.0),
            )
            dst_neighbor_neg_effect = self.to_float_torch(
                dst_padded_node_neighbor_ids.copy(),
                lambda neighbor_id: neg_effect.get(neighbor_id, 0.0),
            )

            src_padded_node_sign_effect = torch.stack(
                [src_neighbor_pos_effect, src_neighbor_neg_effect], dim=1
            )
            dst_padded_node_sign_effect = torch.stack(
                [dst_neighbor_pos_effect, dst_neighbor_neg_effect], dim=1
            )

            # ---- 修复 2026-09-09：RAE = direct 证据（论文 §3.3.2）----
            # 历史位置中 neighbor == 对方(dst_id/src_id) 的位置，按【历史符号】直写 [1,0]/[0,1]
            # 一单位证据（不含当前标签 → 无泄漏）；与 indirect 相加进同一 [pos,neg] 2 通道。
            # 复用 neighbor_sign_effect_layer，无新增参数。RAE 关 ⇒ direct 项为 0。
            if self.module_repeat_aware_sign_encoder:
                src_direct = torch.zeros_like(src_padded_node_sign_effect)
                dst_direct = torch.zeros_like(dst_padded_node_sign_effect)
                # src 侧：seq(u) 中 id==dst_id 的历史位置（u 与 v 的直接历史）
                src_mask = src_padded_node_neighbor_ids == dst_id
                src_mask[0] = False  # 排除 pos0（自身，非历史）
                if src_mask.any():
                    p_idx = np.where(src_mask & (src_padded_node_neighbor_sign == 1))[0]
                    n_idx = np.where(src_mask & (src_padded_node_neighbor_sign == -1))[0]
                    src_direct[p_idx, 0] = 1.0
                    src_direct[n_idx, 1] = 1.0
                # dst 侧：seq(v) 中 id==src_id 的历史位置
                dst_mask = dst_padded_node_neighbor_ids == src_id
                dst_mask[0] = False
                if dst_mask.any():
                    p_idx = np.where(dst_mask & (dst_padded_node_neighbor_sign == 1))[0]
                    n_idx = np.where(dst_mask & (dst_padded_node_neighbor_sign == -1))[0]
                    dst_direct[p_idx, 0] = 1.0
                    dst_direct[n_idx, 1] = 1.0
                src_padded_node_sign_effect = src_padded_node_sign_effect + src_direct
                dst_padded_node_sign_effect = dst_padded_node_sign_effect + dst_direct

            src_padded_nodes_sign_effect.append(src_padded_node_sign_effect)
            dst_padded_nodes_sign_effect.append(dst_padded_node_sign_effect)

            # Tensor, shape (batch_size, src_max_seq_length, 2)
        src_padded_nodes_sign_effect = torch.stack(src_padded_nodes_sign_effect, dim=0)
        # Tensor, shape (batch_size, dst_max_seq_length, 2)
        dst_padded_nodes_sign_effect = torch.stack(dst_padded_nodes_sign_effect, dim=0)

        # set the appearances of the padded node (with zero index) to zeros
        # Tensor, shape (batch_size, src_max_seq_length, 2)
        src_padded_nodes_sign_effect[
            torch.from_numpy(src_padded_nodes_neighbor_ids == 0)
        ] = 0.0
        # Tensor, shape (batch_size, dst_max_seq_length, 2)
        dst_padded_nodes_sign_effect[
            torch.from_numpy(dst_padded_nodes_neighbor_ids == 0)
        ] = 0.0

        return src_padded_nodes_sign_effect, dst_padded_nodes_sign_effect

    def count_nodes_appearances(
        self,
        *,
        src_padded_nodes_neighbor_ids: np.ndarray,
        dst_padded_nodes_neighbor_ids: np.ndarray,
    ):
        """
        count the appearances of nodes in the sequences of source and destination nodes
        改进计数方法
        :param src_padded_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_padded_nodes_neighbor_ids:: ndarray, shape (batch_size, dst_max_seq_length)
        :return:
        """
        # ---- accel 接缝（默认启用；--no-accel / SIGNDYG_ACCEL=0 关闭）----
        # numpy 全批次向量化（2026-09-14 落地，用户批准）；与下方原路径逐位一致
        # （自检：tools/verify/test_cn_vec_seam.py；分解：tools/verify/cn_microbench.py）。
        # K3（Rust 内核，Perf）到货后由同一接缝替换。
        if _accel.on:
            src_app_np, dst_app_np = _accel.cn_counts_vec(
                src_padded_nodes_neighbor_ids, dst_padded_nodes_neighbor_ids
            )
            return (
                torch.from_numpy(src_app_np).to(self.device),
                torch.from_numpy(dst_app_np).to(self.device),
            )
        # two lists to store the appearances of source and destination nodes
        src_padded_nodes_appearances, dst_padded_nodes_appearances = [], []
        # 对每个节点对（单个批次的每个节点）
        # src_padded_node_neighbor_ids, ndarray, shape (src_max_seq_length, )
        # dst_padded_node_neighbor_ids, ndarray, shape (dst_max_seq_length, )
        for (
            src_padded_node_neighbor_ids,
            dst_padded_node_neighbor_ids,
        ) in zip(
            src_padded_nodes_neighbor_ids,
            dst_padded_nodes_neighbor_ids,
        ):

            # src_unique_keys, ndarray, shape (num_src_unique_keys, )
            # src_inverse_indices, ndarray, shape (src_max_seq_length, )
            # src_counts, ndarray, shape (num_src_unique_keys, )
            # we can use src_unique_keys[src_inverse_indices] to reconstruct the original input, and use src_counts[src_inverse_indices] to get counts of the original input
            # 去重邻居数量，反向索引，各个节点计数
            src_unique_keys, src_inverse_indices, src_counts = np.unique(
                src_padded_node_neighbor_ids, return_inverse=True, return_counts=True
            )
            # 将各个邻居的计数还原到邻居的位置上
            # Tensor, shape (src_max_seq_length, )
            src_padded_node_neighbor_counts_in_src = (
                torch.from_numpy(src_counts[src_inverse_indices])
                .float()
                .to(self.device)
            )
            # dictionary, store the mapping relation from unique neighbor id to its appearances for the source node
            # 邻居到邻居出现次数的映射
            src_mapping_dict = dict(zip(src_unique_keys, src_counts))

            # dst_unique_keys, ndarray, shape (num_dst_unique_keys, )
            # dst_inverse_indices, ndarray, shape (dst_max_seq_length, )
            # dst_counts, ndarray, shape (num_dst_unique_keys, )
            # we can use dst_unique_keys[dst_inverse_indices] to reconstruct the original input, and use dst_counts[dst_inverse_indices] to get counts of the original input
            dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(
                dst_padded_node_neighbor_ids, return_inverse=True, return_counts=True
            )
            # Tensor, shape (dst_max_seq_length, )
            dst_padded_node_neighbor_counts_in_dst = (
                torch.from_numpy(dst_counts[dst_inverse_indices])
                .float()
                .to(self.device)
            )
            # dictionary, store the mapping relation from unique neighbor id to its appearances for the destination node
            dst_mapping_dict = dict(zip(dst_unique_keys, dst_counts))

            # we need to use copy() to avoid the modification of src_padded_node_neighbor_ids
            # Tensor, shape (src_max_seq_length, )
            # src 节点的历史邻居在dst里面的出现
            src_padded_node_neighbor_counts_in_dst = self.to_float_torch(
                src_padded_node_neighbor_ids.copy(),
                lambda neighbor_id: dst_mapping_dict.get(neighbor_id, 0.0),
            )

            # Tensor, shape (src_max_seq_length, 2)
            src_padded_nodes_appearances.append(
                torch.stack(
                    [
                        src_padded_node_neighbor_counts_in_src,
                        src_padded_node_neighbor_counts_in_dst,
                    ],
                    dim=1,
                )
            )

            # we need to use copy() to avoid the modification of dst_padded_node_neighbor_ids
            # Tensor, shape (dst_max_seq_length, )
            dst_padded_node_neighbor_counts_in_src = self.to_float_torch(
                dst_padded_node_neighbor_ids.copy(),
                lambda neighbor_id: src_mapping_dict.get(neighbor_id, 0.0),
            )

            # Tensor, shape (dst_max_seq_length, 2)
            dst_padded_nodes_appearances.append(
                torch.stack(
                    [
                        dst_padded_node_neighbor_counts_in_src,
                        dst_padded_node_neighbor_counts_in_dst,
                    ],
                    dim=1,
                )
            )

        # Tensor, shape (batch_size, src_max_seq_length, 2)
        src_padded_nodes_appearances = torch.stack(src_padded_nodes_appearances, dim=0)
        # Tensor, shape (batch_size, dst_max_seq_length, 2)
        dst_padded_nodes_appearances = torch.stack(dst_padded_nodes_appearances, dim=0)

        # set the appearances of the padded node (with zero index) to zeros
        # Tensor, shape (batch_size, src_max_seq_length, 2)
        src_padded_nodes_appearances[
            torch.from_numpy(src_padded_nodes_neighbor_ids == 0)
        ] = 0.0
        # Tensor, shape (batch_size, dst_max_seq_length, 2)
        dst_padded_nodes_appearances[
            torch.from_numpy(dst_padded_nodes_neighbor_ids == 0)
        ] = 0.0

        return src_padded_nodes_appearances, dst_padded_nodes_appearances

    def node_sign_effect_mapping(self, x):
        mask = x.abs().sum(dim=[1, 2]) != 0  # [B]  非零行
        x_nonzero = x[mask]  # [N, F]
        out_nonzero = self.neighbor_sign_effect_layer(x_nonzero)  # LeakyReLU+Linear
        # 把结果填回全零张量
        out = torch.zeros(x.size(0), x.size(1), out_nonzero.size(2), device=x.device)
        out[mask] = out_nonzero

        return out

    def forward(
        self,
        src_ids: np.ndarray,
        dst_ids: np.ndarray,
        src_padded_nodes_neighbor_ids: np.ndarray,
        dst_padded_nodes_neighbor_ids: np.ndarray,
        src_padded_nodes_neighbor_sign: np.ndarray,
        dst_padded_nodes_neighbor_sign: np.ndarray,
        *,
        sample_type: EncodeType,
        node_interact_times: Optional[np.ndarray] = None,
        src_padded_nodes_neighbor_times: Optional[np.ndarray] = None,
        dst_padded_nodes_neighbor_times: Optional[np.ndarray] = None,
    ):
        """
        compute the neighbor co-occurrence features of nodes in src_padded_nodes_neighbor_ids and dst_padded_nodes_neighbor_ids
        :param src_padded_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_padded_nodes_neighbor_ids:: ndarray, shape (batch_size, dst_max_seq_length)
        :return:
        """

        # src_padded_nodes_appearances, Tensor, shape (batch_size, src_max_seq_length, 2)
        # dst_padded_nodes_appearances, Tensor, shape (batch_size, dst_max_seq_length, 2)
        if sample_type == EncodeType.CoOccurredNeighbor:
            # ---- CNE-off 探针（2026-09-17 用户批准；实现选项 (a) 通道置零）----
            # 维度/参数结构不变，仅切断共现信息通道；默认开启 ⇒ 零行为变更；
            # 关闭时结果名自动加 .CNE-D（utils/load_configs.py::result_save_name）。
            if not self.module_common_neighbor_encoder:
                src_cne_zero = torch.zeros(
                    src_padded_nodes_neighbor_ids.shape[0],
                    src_padded_nodes_neighbor_ids.shape[1],
                    self.neighbor_co_occurrence_feat_dim,
                    device=self.device,
                )
                dst_cne_zero = torch.zeros(
                    dst_padded_nodes_neighbor_ids.shape[0],
                    dst_padded_nodes_neighbor_ids.shape[1],
                    self.neighbor_co_occurrence_feat_dim,
                    device=self.device,
                )
                return src_cne_zero, dst_cne_zero
            src_padded_nodes_appearances, dst_padded_nodes_appearances = (
                self.count_nodes_appearances(
                    src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                    dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                )
            )

            # sum the neighbor co-occurrence features in the sequence of source and destination nodes
            # Tensor, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
            src_padded_nodes_neighbor_co_occurrence_features = (
                self.neighbor_co_occurrence_encode_layer(
                    src_padded_nodes_appearances.unsqueeze(dim=-1)
                ).sum(dim=2)
            )
            # Tensor, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
            dst_padded_nodes_neighbor_co_occurrence_features = (
                self.neighbor_co_occurrence_encode_layer(
                    dst_padded_nodes_appearances.unsqueeze(dim=-1)
                ).sum(dim=2)
            )

            return (
                src_padded_nodes_neighbor_co_occurrence_features,
                dst_padded_nodes_neighbor_co_occurrence_features,
            )

        # 采样类似为交互符号编码 且 启用了平衡理论编码器
        elif sample_type == EncodeType.InteractSignEffect:
            if self.module_balance_theory_encoder:
                src_padded_nodes_sign_effect, dst_padded_nodes_sign_effect = (
                    self.count_neighbor_sign_effect(
                        src_nodes=src_ids,
                        dst_nodes=dst_ids,
                        src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                        dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                        src_padded_nodes_neighbor_sign=src_padded_nodes_neighbor_sign,
                        dst_padded_nodes_neighbor_sign=dst_padded_nodes_neighbor_sign,
                        node_interact_times=node_interact_times,
                        src_padded_nodes_neighbor_times=src_padded_nodes_neighbor_times,
                        dst_padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times,
                    )
                )

                src_padded_nodes_sign_effect_features = self.node_sign_effect_mapping(
                    src_padded_nodes_sign_effect
                )
                dst_padded_nodes_sign_effect_features = self.node_sign_effect_mapping(
                    dst_padded_nodes_sign_effect
                )

                return (
                    src_padded_nodes_sign_effect_features,
                    dst_padded_nodes_sign_effect_features,
                )
            else:
                # 未启用，返回空
                return (None, None)
        else:
            raise TypeError("未知采样类型")
