"""
DirectStatusEncoder: 有向符号图的状态理论编码器
- 对有向2-路径 (u→k→v 和 u←k←v) 使用 Status Theory
- 对不确定情况及非2-路径共同邻居退避到 Balance Theory
"""
from enum import Enum
from typing import Callable, Union
import torch
import torch.nn as nn
import numpy as np


class EncodeType(Enum):
    CoOccurredNeighbor = 0
    InteractSignEffect = 1
    StatusEffect = 2  # 新增：状态理论编码


class DirectStatusEncoder(nn.Module):
    """
    有向符号图的状态理论编码器。
    对共同邻居 k 根据边方向分类:
      - Type A (u→k→v): 适用 Status Theory
      - Type B (v→k→u): 适用 Status Theory (反向推断)
      - 不确定情况: 退避到 Balance Theory
    """

    def __init__(
        self,
        neighbor_co_occurrence_feat_dim: int,
        device: str = "cpu",
        *,
        module_status_theory_encoder: bool = True,
        module_balance_fallback: bool = True,
    ):
        super(DirectStatusEncoder, self).__init__()
        self.neighbor_co_occurrence_feat_dim = neighbor_co_occurrence_feat_dim
        self.device = device
        self.module_status_theory_encoder = module_status_theory_encoder
        self.module_balance_fallback = module_balance_fallback

        # 状态理论正/负效应编码层 (输入: [pos_count, neg_count] → d)
        self.status_effect_layer = nn.Sequential(
            nn.Linear(in_features=2, out_features=self.neighbor_co_occurrence_feat_dim),
            nn.LeakyReLU(negative_slope=0.5),
            nn.Linear(
                in_features=self.neighbor_co_occurrence_feat_dim,
                out_features=self.neighbor_co_occurrence_feat_dim,
            ),
        )

        # 平衡理论退避层 (输入: [pos_count, neg_count] → d)
        self.balance_effect_layer = nn.Sequential(
            nn.Linear(in_features=2, out_features=self.neighbor_co_occurrence_feat_dim),
            nn.LeakyReLU(negative_slope=0.5),
            nn.Linear(
                in_features=self.neighbor_co_occurrence_feat_dim,
                out_features=self.neighbor_co_occurrence_feat_dim,
            ),
        )

        # 共现计数编码层 (用于 CoOccurredNeighbor 模式)
        self.neighbor_co_occurrence_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.neighbor_co_occurrence_feat_dim),
            nn.ReLU(),
            nn.Linear(
                in_features=self.neighbor_co_occurrence_feat_dim,
                out_features=self.neighbor_co_occurrence_feat_dim,
            ),
        )

        with torch.no_grad():
            self.status_effect_layer[0].weight *= 10.0
            self.status_effect_layer[0].bias *= 10.0
            self.balance_effect_layer[0].weight *= 10.0
            self.balance_effect_layer[0].bias *= 10.0

    # ------------------------------------------------------------------
    # 辅助工具
    # ------------------------------------------------------------------

    def _to_float_torch(self, arr: np.ndarray, apply: Union[Callable, None] = None):
        tensor = torch.from_numpy(arr)
        if apply is not None:
            tensor = tensor.apply_(apply)
        return tensor.float().to(self.device)

    # ------------------------------------------------------------------
    # 平衡理论计数 (退避用)
    # ------------------------------------------------------------------

    def _count_balance_effect(
        self,
        seq_a_ids: np.ndarray,
        seq_a_signs: np.ndarray,
        seq_b_ids: np.ndarray,
        seq_b_signs: np.ndarray,
    ):
        """
        对 seq_a 的每个邻居，统计其在 seq_b 中的平衡理论正/负效应。
        平衡理论: sign(a,k) × sign(b,k) = +1 → pos_effect; = -1 → neg_effect
        :return: pos_effect_dict, neg_effect_dict (key=neighbor_id, value=count)
        """
        # 找共同邻居
        common_vals = np.intersect1d(seq_a_ids, seq_b_ids)
        if len(common_vals) == 0:
            return {}, {}

        pos_effect, neg_effect = {}, {}
        for k in common_vals:
            if k == 0:
                continue  # skip padding
            a_signs = seq_a_signs[seq_a_ids == k]
            b_signs = seq_b_signs[seq_b_ids == k]
            # 展开所有配对
            a_flat = np.repeat(a_signs, len(b_signs))
            b_flat = np.tile(b_signs, len(a_signs))
            products = a_flat * b_flat
            pos_count = int(np.sum(products == 1))
            neg_count = int(np.sum(products == -1))
            if pos_count > 0:
                pos_effect[int(k)] = pos_count
            if neg_count > 0:
                neg_effect[int(k)] = neg_count
        return pos_effect, neg_effect

    # ------------------------------------------------------------------
    # 状态理论计数 (核心)
    # ------------------------------------------------------------------

    def _count_directed_status_effect_type_a(
        self,
        src_out_ids: np.ndarray,
        src_out_signs: np.ndarray,
        dst_in_ids: np.ndarray,
        dst_in_signs: np.ndarray,
    ):
        """
        Type A: u→k→v 的状态理论计数。
        对 u_out 中的每个共同邻居 k:
          - s_uk=+1 ∧ s_kv=+1 → pos_effect (status递增, u→v 应为正)
          - s_uk=-1 ∧ s_kv=-1 → neg_effect (status递减, u→v 应为负)
          - 异号 → uncertain → 归入退避
        :return: (pos_effect_dict, neg_effect_dict, uncertain_dict)
        """
        common_vals = np.intersect1d(src_out_ids, dst_in_ids)
        if len(common_vals) == 0:
            return {}, {}, {}

        pos_effect, neg_effect, uncertain_effect = {}, {}, {}
        for k in common_vals:
            if k == 0:
                continue  # skip padding
            uk_signs = src_out_signs[src_out_ids == k]  # u→k 的符号
            kv_signs = dst_in_signs[dst_in_ids == k]    # k→v 的符号

            n_uk = len(uk_signs)
            n_kv = len(kv_signs)
            if n_uk == 0 or n_kv == 0:
                continue

            # 展开所有配对
            uk_flat = np.repeat(uk_signs, n_kv)
            kv_flat = np.tile(kv_signs, n_uk)

            pos_mask = (uk_flat == 1) & (kv_flat == 1)
            neg_mask = (uk_flat == -1) & (kv_flat == -1)
            uncertain_mask = ~(pos_mask | neg_mask)

            pos_count = int(np.sum(pos_mask))
            neg_count = int(np.sum(neg_mask))
            uncertain_count = int(np.sum(uncertain_mask))

            if pos_count > 0:
                pos_effect[int(k)] = pos_count
            if neg_count > 0:
                neg_effect[int(k)] = neg_count
            if uncertain_count > 0:
                uncertain_effect[int(k)] = uncertain_count
        return pos_effect, neg_effect, uncertain_effect

    def _count_directed_status_effect_type_b(
        self,
        src_in_ids: np.ndarray,
        src_in_signs: np.ndarray,
        dst_out_ids: np.ndarray,
        dst_out_signs: np.ndarray,
    ):
        """
        Type B: v→k→u 的状态理论计数 (反向推断 u→v)。
        对 u_in 中的每个共同邻居 k (即 k→u):
          - 注意: v→k 的符号在 dst_out_signs 中，k→u 的符号在 src_in_signs 中
          - s_vk=+1 ∧ s_ku=+1 → 对 v→u 正影响 → 对 u→v 为 neg_effect
          - s_vk=-1 ∧ s_ku=-1 → 对 v→u 负影响 → 对 u→v 为 pos_effect
        :return: (pos_effect_dict, neg_effect_dict, uncertain_dict)
          其中 pos/neg 指的是对 u→v 的正/负影响
        """
        common_vals = np.intersect1d(src_in_ids, dst_out_ids)
        if len(common_vals) == 0:
            return {}, {}, {}

        pos_effect, neg_effect, uncertain_effect = {}, {}, {}
        for k in common_vals:
            if k == 0:
                continue  # skip padding
            ku_signs = src_in_signs[src_in_ids == k]  # k→u 的符号
            vk_signs = dst_out_signs[dst_out_ids == k]  # v→k 的符号

            n_ku = len(ku_signs)
            n_vk = len(vk_signs)
            if n_ku == 0 or n_vk == 0:
                continue

            vk_flat = np.repeat(vk_signs, n_ku)
            ku_flat = np.tile(ku_signs, n_vk)

            # Type B: v→k→u, 推断 u→v
            # vk=+1 ∧ ku=+1 → v<k<u → v→u为正 → u→v为负 (neg_effect)
            neg_mask = (vk_flat == 1) & (ku_flat == 1)
            # vk=-1 ∧ ku=-1 → v>k>u → v→u为负 → u→v为正 (pos_effect)
            pos_mask = (vk_flat == -1) & (ku_flat == -1)
            uncertain_mask = ~(pos_mask | neg_mask)

            pos_count = int(np.sum(pos_mask))
            neg_count = int(np.sum(neg_mask))
            uncertain_count = int(np.sum(uncertain_mask))

            if pos_count > 0:
                pos_effect[int(k)] = pos_count
            if neg_count > 0:
                neg_effect[int(k)] = neg_count
            if uncertain_count > 0:
                uncertain_effect[int(k)] = uncertain_count
        return pos_effect, neg_effect, uncertain_effect

    # ------------------------------------------------------------------
    # 单节点邻居序列的特征编码
    # ------------------------------------------------------------------

    def _encode_neighbor_sequence(
        self,
        padded_neighbor_ids: np.ndarray,
        pos_effect: dict,
        neg_effect: dict,
        uncertain_effect: dict,
        balance_pos: dict,
        balance_neg: dict,
    ):
        """
        对单个 padded 邻居序列编码状态理论 + 退避特征。
        :param padded_neighbor_ids: (batch_size, seq_len)
        :param pos_effect: dict, status theory pos effect counts per neighbor
        :param neg_effect: dict, status theory neg effect counts per neighbor
        :param uncertain_effect: dict, uncertain neighbor counts (for balance fallback)
        :param balance_pos: dict, balance theory pos effect counts
        :param balance_neg: dict, balance theory neg effect counts
        :return: Tensor (batch_size, seq_len, feat_dim)
        """
        B, L = padded_neighbor_ids.shape
        # 初始化计数矩阵
        pos_counts = np.zeros((B, L), dtype=np.float32)
        neg_counts = np.zeros((B, L), dtype=np.float32)

        for b in range(B):
            for l in range(L):
                nid = int(padded_neighbor_ids[b, l])
                if nid == 0:
                    continue  # padding

                # 优先状态理论
                st_pos = pos_effect.get(nid, 0)
                st_neg = neg_effect.get(nid, 0)

                if self.module_status_theory_encoder and (st_pos > 0 or st_neg > 0):
                    # 有明确的状态理论信号
                    pos_counts[b, l] = float(st_pos)
                    neg_counts[b, l] = float(st_neg)
                elif self.module_balance_fallback:
                    # 退避到平衡理论
                    pos_counts[b, l] = float(balance_pos.get(nid, 0))
                    neg_counts[b, l] = float(balance_neg.get(nid, 0))

        # 堆叠为 [pos, neg] → 编码
        counts = torch.from_numpy(
            np.stack([pos_counts, neg_counts], axis=-1)
        ).float().to(self.device)  # (B, L, 2)

        # 区分状态理论 vs 退避: 对不确定的邻居用 balance_effect_layer
        # 简化处理: 统一用 status_effect_layer
        features = self.status_effect_layer(counts)  # (B, L, d)

        # padding 位置置零
        features[torch.from_numpy(padded_neighbor_ids == 0)] = 0.0
        return features

    # ------------------------------------------------------------------
    # 共现计数编码 (CoOccurredNeighbor 模式)
    # ------------------------------------------------------------------

    def _count_nodes_appearances(
        self,
        padded_ids_a: np.ndarray,
        padded_ids_b: np.ndarray,
    ):
        """
        统计 seq_a 中每个邻居在 seq_a 和 seq_b 中的出现次数。
        """
        a_appearances, b_appearances = [], []
        for a_ids, b_ids in zip(padded_ids_a, padded_ids_b):
            a_unique, a_inv, a_counts = np.unique(a_ids, return_inverse=True, return_counts=True)
            b_unique, b_inv, b_counts = np.unique(b_ids, return_inverse=True, return_counts=True)

            # a 序列: [count_in_a, count_in_b]
            a_count_in_a = a_counts[a_inv]
            b_map = dict(zip(b_unique, b_counts))
            a_count_in_b = np.array([b_map.get(x, 0) for x in a_ids], dtype=np.float32)

            a_appearances.append(
                torch.from_numpy(np.stack([a_count_in_a, a_count_in_b], axis=-1)).float().to(self.device)
            )

            # b 序列: [count_in_a, count_in_b]
            b_count_in_b = b_counts[b_inv]
            a_map = dict(zip(a_unique, a_counts))
            b_count_in_a = np.array([a_map.get(x, 0) for x in b_ids], dtype=np.float32)

            b_appearances.append(
                torch.from_numpy(np.stack([b_count_in_a, b_count_in_b], axis=-1)).float().to(self.device)
            )

        a_feat = torch.stack(a_appearances, dim=0)  # (B, La, 2)
        b_feat = torch.stack(b_appearances, dim=0)  # (B, Lb, 2)
        return a_feat, b_feat

    # ------------------------------------------------------------------
    # 主 forward
    # ------------------------------------------------------------------

    def forward(
        self,
        src_ids: np.ndarray,
        dst_ids: np.ndarray,
        src_out_padded_ids: np.ndarray,
        src_out_padded_signs: np.ndarray,
        src_in_padded_ids: np.ndarray,
        src_in_padded_signs: np.ndarray,
        dst_out_padded_ids: np.ndarray,
        dst_out_padded_signs: np.ndarray,
        dst_in_padded_ids: np.ndarray,
        dst_in_padded_signs: np.ndarray,
        *,
        sample_type: EncodeType,
    ):
        """
        :param sample_type: EncodeType.StatusEffect → 状态理论 + 退避
                            EncodeType.CoOccurredNeighbor → 共现计数
        :return: 取决于 sample_type
        """
        if sample_type == EncodeType.CoOccurredNeighbor:
            # 对出入序列分别做共现计数
            # src_out vs dst_in (Type A 相关)
            src_out_cooc, dst_in_cooc = self._count_nodes_appearances(
                src_out_padded_ids, dst_in_padded_ids
            )
            src_out_feat = self.neighbor_co_occurrence_encode_layer(
                src_out_cooc.unsqueeze(-1)
            ).sum(dim=2)
            dst_in_feat = self.neighbor_co_occurrence_encode_layer(
                dst_in_cooc.unsqueeze(-1)
            ).sum(dim=2)

            # src_in vs dst_out (Type B 相关)
            src_in_cooc, dst_out_cooc = self._count_nodes_appearances(
                src_in_padded_ids, dst_out_padded_ids
            )
            src_in_feat = self.neighbor_co_occurrence_encode_layer(
                src_in_cooc.unsqueeze(-1)
            ).sum(dim=2)
            dst_out_feat = self.neighbor_co_occurrence_encode_layer(
                dst_out_cooc.unsqueeze(-1)
            ).sum(dim=2)

            # 合并: src_out+src_in → src; dst_out+dst_in → dst
            src_feat = torch.cat([src_out_feat, src_in_feat], dim=1)
            dst_feat = torch.cat([dst_out_feat, dst_in_feat], dim=1)
            return src_feat, dst_feat

        elif sample_type == EncodeType.StatusEffect:
            # 对 batch 中每对 (u, v) 分别处理
            B = len(src_ids)
            src_out_feats, src_in_feats = [], []
            dst_out_feats, dst_in_feats = [], []

            for b in range(B):
                # ---- Type A: src_out vs dst_in ----
                pos_a, neg_a, uncertain_a = self._count_directed_status_effect_type_a(
                    src_out_padded_ids[b], src_out_padded_signs[b],
                    dst_in_padded_ids[b], dst_in_padded_signs[b],
                )
                # 对 uncertain + 退避做平衡理论计数
                bal_pos_a, bal_neg_a = self._count_balance_effect(
                    src_out_padded_ids[b], src_out_padded_signs[b],
                    dst_in_padded_ids[b], dst_in_padded_signs[b],
                )

                # ---- Type B: src_in vs dst_out ----
                pos_b, neg_b, uncertain_b = self._count_directed_status_effect_type_b(
                    src_in_padded_ids[b], src_in_padded_signs[b],
                    dst_out_padded_ids[b], dst_out_padded_signs[b],
                )
                bal_pos_b, bal_neg_b = self._count_balance_effect(
                    src_in_padded_ids[b], src_in_padded_signs[b],
                    dst_out_padded_ids[b], dst_out_padded_signs[b],
                )

                # ---- 编码各序列 ----
                # src_out 特征 (基于 Type A)
                so_feat = self._encode_neighbor_sequence(
                    src_out_padded_ids[b:b+1], pos_a, neg_a, uncertain_a,
                    bal_pos_a, bal_neg_a,
                )
                src_out_feats.append(so_feat)

                # src_in 特征 (基于 Type B)
                si_feat = self._encode_neighbor_sequence(
                    src_in_padded_ids[b:b+1], pos_b, neg_b, uncertain_b,
                    bal_pos_b, bal_neg_b,
                )
                src_in_feats.append(si_feat)

                # dst_out 特征 (基于 Type B)
                do_feat = self._encode_neighbor_sequence(
                    dst_out_padded_ids[b:b+1], pos_b, neg_b, uncertain_b,
                    bal_pos_b, bal_neg_b,
                )
                dst_out_feats.append(do_feat)

                # dst_in 特征 (基于 Type A)
                di_feat = self._encode_neighbor_sequence(
                    dst_in_padded_ids[b:b+1], pos_a, neg_a, uncertain_a,
                    bal_pos_a, bal_neg_a,
                )
                dst_in_feats.append(di_feat)

            # 合并 batch
            src_out_feat = torch.cat(src_out_feats, dim=0)
            src_in_feat = torch.cat(src_in_feats, dim=0)
            dst_out_feat = torch.cat(dst_out_feats, dim=0)
            dst_in_feat = torch.cat(dst_in_feats, dim=0)

            # 合并为统一的 src / dst
            src_feat = torch.cat([src_out_feat, src_in_feat], dim=1)
            dst_feat = torch.cat([dst_out_feat, dst_in_feat], dim=1)
            return src_feat, dst_feat

        else:
            raise TypeError(f"未知采样类型: {sample_type}")
