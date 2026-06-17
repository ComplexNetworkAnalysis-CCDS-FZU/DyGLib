"""
DirectSignDyGFormer: 有向符号图 DyGFormer
- 每个节点维护独立的出邻居 (u→k) 和入邻居 (k→u) 序列
- 使用 Status Theory (状态理论) 编码有向2-路径
- 不确定情况退避到 Balance Theory
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from models.modules import AutoClassName, TimeEncoder
from models.DirectStatusEncoder import DirectStatusEncoder, EncodeType as DirectEncodeType
from utils.direct_neighbor_sampler import DirectedNeighborSampler as NeighborSampler

from utils.profiler import Profiler


class DirectSignDyGFormer(nn.Module, metaclass=AutoClassName):

    def __init__(
        self,
        node_raw_features: np.ndarray,
        edge_raw_features: np.ndarray,
        neighbor_sampler: NeighborSampler,
        time_feat_dim: int,
        channel_embedding_dim: int,
        patch_size: int = 1,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        max_input_sequence_length: int = 512,
        device: str = "cpu",
        *,
        module_status_theory_encoder: bool = True,
        module_balance_fallback: bool = True,
    ):
        """
        DirectSignDyGFormer: DyGFormer 的有向符号图版本。
        :param module_status_theory_encoder: 是否启用状态理论编码通道
        :param module_balance_fallback: 不确定时是否退避到平衡理论
        """
        super(DirectSignDyGFormer, self).__init__()

        self.module_status_theory_encoder = module_status_theory_encoder

        self.node_raw_features = torch.from_numpy(
            node_raw_features.astype(np.float32)
        ).to(device)
        self.edge_raw_features = torch.from_numpy(
            edge_raw_features.astype(np.float32)
        ).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_input_sequence_length = max_input_sequence_length
        self.device = device

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim

        # 有向状态理论编码器 (替代 NeighborCooccurrenceEncoder)
        self.status_encoder = DirectStatusEncoder(
            neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim,
            device=self.device,
            module_status_theory_encoder=module_status_theory_encoder,
            module_balance_fallback=module_balance_fallback,
        )

        self.projection_layer = nn.ModuleDict(
            {
                "node": nn.Linear(
                    in_features=self.patch_size * self.node_feat_dim,
                    out_features=self.channel_embedding_dim,
                    bias=True,
                ),
                "edge": nn.Linear(
                    in_features=self.patch_size * self.edge_feat_dim,
                    out_features=self.channel_embedding_dim,
                    bias=True,
                ),
                "time": nn.Linear(
                    in_features=self.patch_size * self.time_feat_dim,
                    out_features=self.channel_embedding_dim,
                    bias=True,
                ),
                "neighbor_co_occurrence": nn.Linear(
                    in_features=self.patch_size * self.neighbor_co_occurrence_feat_dim,
                    out_features=self.channel_embedding_dim,
                    bias=True,
                ),
                "status_effect": nn.Linear(
                    in_features=self.patch_size * self.neighbor_co_occurrence_feat_dim,
                    out_features=self.channel_embedding_dim,
                    bias=True,
                ),
            }
        )

        self.num_channels = 5 if self.module_status_theory_encoder else 4

        self.transformers = nn.ModuleList(
            [
                TransformerEncoder(
                    attention_dim=self.num_channels * self.channel_embedding_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.output_layer = nn.Linear(
            in_features=self.num_channels * self.channel_embedding_dim,
            out_features=self.node_feat_dim,
            bias=True,
        )

        self.profiler = Profiler()
        self.profiler.disable()

    # ------------------------------------------------------------------
    # 主前向: 计算 src/dst 节点时序嵌入
    # ------------------------------------------------------------------

    def compute_src_dst_node_temporal_embeddings(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        node_interact_sign: np.ndarray,
    ):
        """
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param node_interact_sign: ndarray, shape (batch_size, )
        """
        pf = self.profiler

        # ---- Step 1: Directed History Sampling ----
        with pf.timer("History Sampling (Directed)"):
            (
                src_out_ids_list, src_out_edges_list, src_out_times_list, src_out_signs_list,
                src_in_ids_list,  src_in_edges_list,  src_in_times_list,  src_in_signs_list,
                dst_out_ids_list, dst_out_edges_list, dst_out_times_list, dst_out_signs_list,
                dst_in_ids_list,  dst_in_edges_list,  dst_in_times_list,  dst_in_signs_list,
            ) = self.neighbor_sampler.history_neighbors_sampling_directed(
                src_node_ids, dst_node_ids, node_interact_times
            )

        # ---- Step 2: Sequence Padding (4 序列) ----
        with pf.timer("Sequence Padding"):
            (src_out_padded_ids, src_out_padded_edges, src_out_padded_times, src_out_padded_signs,
             ) = self._pad_sequences(
                node_ids=src_node_ids,
                node_interact_times=node_interact_times,
                nodes_neighbor_ids_list=src_out_ids_list,
                nodes_edge_ids_list=src_out_edges_list,
                nodes_neighbor_times_list=src_out_times_list,
                node_interact_sign=node_interact_sign,
                nodes_neighbor_sign_list=src_out_signs_list,
            )
            (src_in_padded_ids, src_in_padded_edges, src_in_padded_times, src_in_padded_signs,
             ) = self._pad_sequences(
                node_ids=src_node_ids,
                node_interact_times=node_interact_times,
                nodes_neighbor_ids_list=src_in_ids_list,
                nodes_edge_ids_list=src_in_edges_list,
                nodes_neighbor_times_list=src_in_times_list,
                node_interact_sign=node_interact_sign,
                nodes_neighbor_sign_list=src_in_signs_list,
            )
            (dst_out_padded_ids, dst_out_padded_edges, dst_out_padded_times, dst_out_padded_signs,
             ) = self._pad_sequences(
                node_ids=dst_node_ids,
                node_interact_times=node_interact_times,
                nodes_neighbor_ids_list=dst_out_ids_list,
                nodes_edge_ids_list=dst_out_edges_list,
                nodes_neighbor_times_list=dst_out_times_list,
                node_interact_sign=node_interact_sign,
                nodes_neighbor_sign_list=dst_out_signs_list,
            )
            (dst_in_padded_ids, dst_in_padded_edges, dst_in_padded_times, dst_in_padded_signs,
             ) = self._pad_sequences(
                node_ids=dst_node_ids,
                node_interact_times=node_interact_times,
                nodes_neighbor_ids_list=dst_in_ids_list,
                nodes_edge_ids_list=dst_in_edges_list,
                nodes_neighbor_times_list=dst_in_times_list,
                node_interact_sign=node_interact_sign,
                nodes_neighbor_sign_list=dst_in_signs_list,
            )

        # ---- Step 3: Co-Occurrence Encoding ----
        with pf.timer("Co-Occurrence Encoding"):
            src_cooc_features, dst_cooc_features = self.status_encoder.forward(
                src_ids=src_node_ids,
                dst_ids=dst_node_ids,
                src_out_padded_ids=src_out_padded_ids,
                src_out_padded_signs=src_out_padded_signs,
                src_in_padded_ids=src_in_padded_ids,
                src_in_padded_signs=src_in_padded_signs,
                dst_out_padded_ids=dst_out_padded_ids,
                dst_out_padded_signs=dst_out_padded_signs,
                dst_in_padded_ids=dst_in_padded_ids,
                dst_in_padded_signs=dst_in_padded_signs,
                sample_type=DirectEncodeType.CoOccurredNeighbor,
            )

        # ---- Step 4: Status Theory Encoding ----
        with pf.timer("Status Theory Encoding"):
            src_status_features, dst_status_features = self.status_encoder.forward(
                src_ids=src_node_ids,
                dst_ids=dst_node_ids,
                src_out_padded_ids=src_out_padded_ids,
                src_out_padded_signs=src_out_padded_signs,
                src_in_padded_ids=src_in_padded_ids,
                src_in_padded_signs=src_in_padded_signs,
                dst_out_padded_ids=dst_out_padded_ids,
                dst_out_padded_signs=dst_out_padded_signs,
                dst_in_padded_ids=dst_in_padded_ids,
                dst_in_padded_signs=dst_in_padded_signs,
                sample_type=DirectEncodeType.StatusEffect,
            )

        # ---- Step 5: Node/Edge/Time Feature Encoding (4 序列) ----
        with pf.timer("Node, Edge and Time Encoding"):
            # src_out
            src_out_node_feat, src_out_edge_feat, src_out_time_feat = self._get_features(
                node_interact_times, src_out_padded_ids, src_out_padded_edges, src_out_padded_times,
            )
            # src_in
            src_in_node_feat, src_in_edge_feat, src_in_time_feat = self._get_features(
                node_interact_times, src_in_padded_ids, src_in_padded_edges, src_in_padded_times,
            )
            # dst_out
            dst_out_node_feat, dst_out_edge_feat, dst_out_time_feat = self._get_features(
                node_interact_times, dst_out_padded_ids, dst_out_padded_edges, dst_out_padded_times,
            )
            # dst_in
            dst_in_node_feat, dst_in_edge_feat, dst_in_time_feat = self._get_features(
                node_interact_times, dst_in_padded_ids, dst_in_padded_edges, dst_in_padded_times,
            )

        # ---- Step 6: Patching (4 序列) ----
        with pf.timer("Patching"):
            # src_out patches
            (so_node_patches, so_edge_patches, so_time_patches) = self._get_patches_simple(
                src_out_node_feat, src_out_edge_feat, src_out_time_feat,
            )
            # src_in patches
            (si_node_patches, si_edge_patches, si_time_patches) = self._get_patches_simple(
                src_in_node_feat, src_in_edge_feat, src_in_time_feat,
            )
            # dst_out patches
            (do_node_patches, do_edge_patches, do_time_patches) = self._get_patches_simple(
                dst_out_node_feat, dst_out_edge_feat, dst_out_time_feat,
            )
            # dst_in patches
            (di_node_patches, di_edge_patches, di_time_patches) = self._get_patches_simple(
                dst_in_node_feat, dst_in_edge_feat, dst_in_time_feat,
            )

            # cooc & status patches — 因为 encoder 返回的是合并的 src/dst，需要按出入拆分
            # 取各半长度作为 out/in 划分（encoder 内部 concat 了 out+in）
            src_out_len = src_out_node_feat.shape[1]
            src_in_len = src_in_node_feat.shape[1]
            dst_out_len = dst_out_node_feat.shape[1]
            dst_in_len = dst_in_node_feat.shape[1]

            so_cooc_patches = self._slice_and_patch(src_cooc_features, src_out_len)
            si_cooc_patches = self._slice_and_patch(
                src_cooc_features[:, src_out_len:, :], src_in_len
            )
            do_cooc_patches = self._slice_and_patch(dst_cooc_features, dst_out_len)
            di_cooc_patches = self._slice_and_patch(
                dst_cooc_features[:, dst_out_len:, :], dst_in_len
            )

            if self.module_status_theory_encoder:
                so_status_patches = self._slice_and_patch(src_status_features, src_out_len)
                si_status_patches = self._slice_and_patch(
                    src_status_features[:, src_out_len:, :], src_in_len
                )
                do_status_patches = self._slice_and_patch(dst_status_features, dst_out_len)
                di_status_patches = self._slice_and_patch(
                    dst_status_features[:, dst_out_len:, :], dst_in_len
                )

        # ---- Step 7: Merge src_out + src_in → src; dst_out + dst_in → dst ----
        with pf.timer("Patch Align & Merge"):
            # 合并 node patches
            src_node_patches = torch.cat([so_node_patches, si_node_patches], dim=1)
            dst_node_patches = torch.cat([do_node_patches, di_node_patches], dim=1)
            # 合并 edge patches
            src_edge_patches = torch.cat([so_edge_patches, si_edge_patches], dim=1)
            dst_edge_patches = torch.cat([do_edge_patches, di_edge_patches], dim=1)
            # 合并 time patches
            src_time_patches = torch.cat([so_time_patches, si_time_patches], dim=1)
            dst_time_patches = torch.cat([do_time_patches, di_time_patches], dim=1)
            # 合并 cooc patches
            src_cooc_patches = torch.cat([so_cooc_patches, si_cooc_patches], dim=1)
            dst_cooc_patches = torch.cat([do_cooc_patches, di_cooc_patches], dim=1)

            if self.module_status_theory_encoder:
                src_status_patches = torch.cat([so_status_patches, si_status_patches], dim=1)
                dst_status_patches = torch.cat([do_status_patches, di_status_patches], dim=1)

            # ---- Projection ----
            src_node_proj = self.projection_layer["node"](src_node_patches)
            dst_node_proj = self.projection_layer["node"](dst_node_patches)
            src_edge_proj = self.projection_layer["edge"](src_edge_patches)
            dst_edge_proj = self.projection_layer["edge"](dst_edge_patches)
            src_time_proj = self.projection_layer["time"](src_time_patches)
            dst_time_proj = self.projection_layer["time"](dst_time_patches)
            src_cooc_proj = self.projection_layer["neighbor_co_occurrence"](src_cooc_patches)
            dst_cooc_proj = self.projection_layer["neighbor_co_occurrence"](dst_cooc_patches)

            if self.module_status_theory_encoder:
                src_status_proj = self.projection_layer["status_effect"](src_status_patches)
                dst_status_proj = self.projection_layer["status_effect"](dst_status_patches)

            # ---- Concat src & dst along patch dim ----
            batch_size = src_node_proj.shape[0]
            src_num_patches = src_node_proj.shape[1]
            dst_num_patches = dst_node_proj.shape[1]

            patches_node = torch.cat([src_node_proj, dst_node_proj], dim=1)
            patches_edge = torch.cat([src_edge_proj, dst_edge_proj], dim=1)
            patches_time = torch.cat([src_time_proj, dst_time_proj], dim=1)
            patches_cooc = torch.cat([src_cooc_proj, dst_cooc_proj], dim=1)

            patches_data = [patches_node, patches_edge, patches_time, patches_cooc]
            if self.module_status_theory_encoder:
                patches_status = torch.cat([src_status_proj, dst_status_proj], dim=1)
                patches_data.append(patches_status)

            # Stack: (B, total_patches, num_channels, channel_embedding_dim)
            patches_data = torch.stack(patches_data, dim=2)
            # Flatten: (B, total_patches, num_channels * channel_embedding_dim)
            patches_data = patches_data.reshape(
                batch_size,
                src_num_patches + dst_num_patches,
                self.num_channels * self.channel_embedding_dim,
            )

        # ---- Step 8: Transformer ----
        with pf.timer("Transformer"):
            for transformer in self.transformers:
                patches_data = transformer(patches_data)

        # ---- Step 9: Split & Output ----
        with pf.timer("Embedding Split"):
            src_patches_data = patches_data[:, :src_num_patches, :]
            dst_patches_data = patches_data[
                :, src_num_patches : src_num_patches + dst_num_patches, :
            ]
            src_patches_data = torch.mean(src_patches_data, dim=1)
            dst_patches_data = torch.mean(dst_patches_data, dim=1)

            src_node_embeddings = self.output_layer(src_patches_data)
            dst_node_embeddings = self.output_layer(dst_patches_data)

        pf.report()
        return src_node_embeddings, dst_node_embeddings

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _pad_sequences(
        self,
        node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        nodes_neighbor_ids_list: list,
        nodes_edge_ids_list: list,
        nodes_neighbor_times_list: list,
        node_interact_sign: np.ndarray,
        nodes_neighbor_sign_list: list,
    ):
        """单个序列的 padding，与 SignDyGFormer.pad_sequences 逻辑一致。"""
        max_seq_length = 0
        for idx in range(len(nodes_neighbor_ids_list)):
            assert (
                len(nodes_neighbor_ids_list[idx])
                == len(nodes_edge_ids_list[idx])
                == len(nodes_neighbor_times_list[idx])
                == len(nodes_neighbor_sign_list[idx])
            )
            if len(nodes_neighbor_ids_list[idx]) > self.max_input_sequence_length - 1:
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][
                    -(self.max_input_sequence_length - 1):
                ]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][
                    -(self.max_input_sequence_length - 1):
                ]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][
                    -(self.max_input_sequence_length - 1):
                ]
                nodes_neighbor_sign_list[idx] = nodes_neighbor_sign_list[idx][
                    -(self.max_input_sequence_length - 1):
                ]
            if len(nodes_neighbor_ids_list[idx]) > max_seq_length:
                max_seq_length = len(nodes_neighbor_ids_list[idx])

        max_seq_length += 1
        if max_seq_length % self.patch_size != 0:
            max_seq_length += self.patch_size - max_seq_length % self.patch_size

        padded_ids = np.zeros((len(node_ids), max_seq_length), dtype=np.longlong)
        padded_edges = np.zeros((len(node_ids), max_seq_length), dtype=np.longlong)
        padded_times = np.zeros((len(node_ids), max_seq_length), dtype=np.float32)
        padded_signs = np.zeros((len(node_ids), max_seq_length), dtype=np.int8)

        for idx in range(len(node_ids)):
            padded_ids[idx, 0] = node_ids[idx]
            padded_edges[idx, 0] = 0
            padded_times[idx, 0] = node_interact_times[idx]
            padded_signs[idx, 0] = node_interact_sign[idx]

            n_neighbors = len(nodes_neighbor_ids_list[idx])
            if n_neighbors > 0:
                padded_ids[idx, 1 : n_neighbors + 1] = nodes_neighbor_ids_list[idx]
                padded_edges[idx, 1 : n_neighbors + 1] = nodes_edge_ids_list[idx]
                padded_times[idx, 1 : n_neighbors + 1] = nodes_neighbor_times_list[idx]
                padded_signs[idx, 1 : n_neighbors + 1] = nodes_neighbor_sign_list[idx]

        return padded_ids, padded_edges, padded_times, padded_signs

    def _get_features(
        self,
        node_interact_times: np.ndarray,
        padded_ids: np.ndarray,
        padded_edges: np.ndarray,
        padded_times: np.ndarray,
    ):
        """获取 node/edge/time 特征。"""
        node_feat = self.node_raw_features[torch.from_numpy(padded_ids)]
        edge_feat = self.edge_raw_features[torch.from_numpy(padded_edges)]
        time_feat = self.time_encoder(
            timestamps=torch.from_numpy(
                node_interact_times[:, np.newaxis] - padded_times
            ).float().to(self.device)
        )
        time_feat[torch.from_numpy(padded_ids == 0)] = 0.0
        return node_feat, edge_feat, time_feat

    def _get_patches_simple(
        self,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor,
        time_feat: torch.Tensor,
    ):
        """对 node/edge/time 三组特征做 patching（不含 cooc/status）。"""
        assert node_feat.shape[1] % self.patch_size == 0
        num_patches = node_feat.shape[1] // self.patch_size
        B = node_feat.shape[0]

        node_patches_list, edge_patches_list, time_patches_list = [], [], []
        for p in range(num_patches):
            s, e = p * self.patch_size, (p + 1) * self.patch_size
            node_patches_list.append(node_feat[:, s:e, :])
            edge_patches_list.append(edge_feat[:, s:e, :])
            time_patches_list.append(time_feat[:, s:e, :])

        node_patches = torch.stack(node_patches_list, dim=1).reshape(
            B, num_patches, self.patch_size * self.node_feat_dim
        )
        edge_patches = torch.stack(edge_patches_list, dim=1).reshape(
            B, num_patches, self.patch_size * self.edge_feat_dim
        )
        time_patches = torch.stack(time_patches_list, dim=1).reshape(
            B, num_patches, self.patch_size * self.time_feat_dim
        )
        return node_patches, edge_patches, time_patches

    def _slice_and_patch(self, feat: torch.Tensor, seq_len: int):
        """将特征切分为 patches。"""
        assert seq_len % self.patch_size == 0, f"seq_len {seq_len} not divisible by patch_size {self.patch_size}"
        num_patches = seq_len // self.patch_size
        B = feat.shape[0]
        patches_list = []
        for p in range(num_patches):
            s, e = p * self.patch_size, (p + 1) * self.patch_size
            patches_list.append(feat[:, s:e, :])
        return torch.stack(patches_list, dim=1).reshape(
            B, num_patches, self.patch_size * self.neighbor_co_occurrence_feat_dim
        )

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in [
            "uniform",
            "time_interval_aware",
        ]:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


class TransformerEncoder(nn.Module):
    """与 SignDyGFormer 中的 TransformerEncoder 完全一致。"""

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()
        self.multi_head_attention = MultiheadAttention(
            embed_dim=attention_dim, num_heads=num_heads, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
                nn.Linear(in_features=4 * attention_dim, out_features=attention_dim),
            ]
        )
        self.norm_layers = nn.ModuleList(
            [nn.LayerNorm(attention_dim), nn.LayerNorm(attention_dim)]
        )

    def forward(self, inputs: torch.Tensor):
        transposed_inputs = inputs.transpose(0, 1)
        transposed_inputs = self.norm_layers[0](transposed_inputs)
        hidden_states = self.multi_head_attention(
            query=transposed_inputs, key=transposed_inputs, value=transposed_inputs
        )[0].transpose(0, 1)
        outputs = inputs + self.dropout(hidden_states)
        hidden_states = self.linear_layers[1](
            self.dropout(F.gelu(self.linear_layers[0](self.norm_layers[1](outputs))))
        )
        outputs = outputs + self.dropout(hidden_states)
        return outputs
