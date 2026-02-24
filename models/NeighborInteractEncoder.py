from enum import Enum
from typing import Callable, Union
from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import MultiheadAttention

class EncodeType(Enum):
    CoOccurredNeighbor = 0
    InteractSignEffect = 1


class NeighborCooccurrenceEncoder(nn.Module):

    def __init__(
        self,
        neighbor_co_occurrence_feat_dim: int,
        device: str = "cpu",
        *,
        pair_sign_effect_aware:bool = False
    ):
        """
        Neighbor co-occurrence encoder.
        :param neighbor_co_occurrence_feat_dim: int, dimension of neighbor co-occurrence features (encodings)
        :param device: str, device
        """
        super(NeighborCooccurrenceEncoder, self).__init__()
        self.neighbor_co_occurrence_feat_dim = neighbor_co_occurrence_feat_dim
        self.device = device
        self.pair_sign_effect_aware= pair_sign_effect_aware

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
    ):
        """
        计算src-dst 之间的共同邻居对符号的影响
        - 使用平衡理论, 对于共同邻居 k, 节点对u,v 有以下情况
          - u + k  k + v = 正边影响 +1
          - u - k  k - v = 正边影响 +1
          - u - k  k + v = 负边影响 +1
          - u + k  k - v = 负边影响 +1

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

        dst_common_neighbor_ids = dst_node_neighbor_ids[dst_common_idx]
        dst_common_neighbor_sign = dst_node_neighbor_sign[dst_common_idx]

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

        #
        pos_effect = src_common_neighbor_ids[src_idx_flat[mask][pos_suggest_mask]]
        neg_effect = src_common_neighbor_ids[src_idx_flat[mask][neg_suggest_mask]]

        pos_keys, pos_count = np.unique(pos_effect, return_counts=True)
        neg_keys, neg_count = np.unique(neg_effect, return_counts=True)
        pos_effect = dict(zip(pos_keys, pos_count))
        neg_effect = dict(zip(neg_keys, neg_count))

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
    ):
        src_padded_nodes_sign_effect, dst_padded_nodes_sign_effect = [], []
        # 对每个节点对（单个批次的每个节点）
        # src_padded_node_neighbor_ids, ndarray, shape (src_max_seq_length, )
        # dst_padded_node_neighbor_ids, ndarray, shape (dst_max_seq_length, )
        for (
            src_id,
            dst_id,
            src_padded_node_neighbor_ids,
            dst_padded_node_neighbor_ids,
            src_padded_node_neighbor_sign,
            dst_padded_node_neighbor_sign,
        ) in zip(
            src_nodes,
            dst_nodes,
            src_padded_nodes_neighbor_ids,
            dst_padded_nodes_neighbor_ids,
            src_padded_nodes_neighbor_sign,
            dst_padded_nodes_neighbor_sign,
        ):
            src_unique_keys = np.unique(
                src_padded_node_neighbor_ids,
            )

            dst_unique_keys = np.unique(
                dst_padded_node_neighbor_ids,
            )

            common_neighbor = np.intersect1d(src_unique_keys, dst_unique_keys)

            if self.pair_sign_effect_aware:
                # 同时感知当前交互节点对的信息
                # 假定节点不会自己和自己交互
                np.append(common_neighbor,[src_id,dst_id])

            pos_effect, neg_effect = self.sign_effect_count(
                common_neighbor=common_neighbor,
                src_node_neighbor_ids=src_padded_node_neighbor_ids,
                src_node_neighbor_sign=src_padded_node_neighbor_sign,
                dst_node_neighbor_ids=dst_padded_node_neighbor_ids,
                dst_node_neighbor_sign=dst_padded_node_neighbor_sign,
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

            src_padded_node_sign_effect = torch.torch.stack(
                [src_neighbor_pos_effect, src_neighbor_neg_effect], dim=1
            )
            dst_padded_node_sign_effect = torch.stack(
                [dst_neighbor_pos_effect, dst_neighbor_neg_effect], dim=1
            )

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
        src_ids:np.ndarray,
        dst_ids:np.ndarray,
        src_padded_nodes_neighbor_ids: np.ndarray,
        dst_padded_nodes_neighbor_ids: np.ndarray,
        src_padded_nodes_neighbor_sign: np.ndarray,
        dst_padded_nodes_neighbor_sign: np.ndarray,
        *,
        sample_type: EncodeType,
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

        if sample_type == EncodeType.InteractSignEffect:
            src_padded_nodes_sign_effect, dst_padded_nodes_sign_effect = (
                self.count_neighbor_sign_effect(
                    src_nodes=src_ids,
                    dst_nodes=dst_ids,
                    src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                    dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                    src_padded_nodes_neighbor_sign=src_padded_nodes_neighbor_sign,
                    dst_padded_nodes_neighbor_sign=dst_padded_nodes_neighbor_sign,
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
            raise TypeError("未知采样类型")

