import typing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from models.modules import TimeEncoder
from utils.utils import NeighborSampler
from typing import Union,Callable

class SignDyGFormer(nn.Module):

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
    ):
        """
        DyGFormer model.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param channel_embedding_dim: int, dimension of each channel embedding
        :param patch_size: int, patch size
        :param num_layers: int, number of transformer layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param max_input_sequence_length: int, maximal length of the input sequence for each node
        :param device: str, device
        """
        super(SignDyGFormer, self).__init__()

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
        self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(
            neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim,
            device=self.device,
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
            }
        )

        self.num_channels = 4

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

    def compute_src_dst_node_temporal_embeddings(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        node_interact_sign: np.ndarray,
    ):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :return:
        """
        # get the first-hop neighbors of source and destination nodes
        # three lists to store source nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        (
            src_nodes_neighbor_ids_list,
            src_nodes_edge_ids_list,
            src_nodes_neighbor_times_list,
            src_nodes_neighbor_sign_list,
        ) = self.neighbor_sampler.get_all_first_hop_neighbors(
            node_ids=src_node_ids, node_interact_times=node_interact_times
        )

        # three lists to store destination nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        (
            dst_nodes_neighbor_ids_list,
            dst_nodes_edge_ids_list,
            dst_nodes_neighbor_times_list,
            dst_nodes_neighbor_sign_list,
        ) = self.neighbor_sampler.get_all_first_hop_neighbors(
            node_ids=dst_node_ids, node_interact_times=node_interact_times
        )

        # pad the sequences of first-hop neighbors for source and destination nodes
        # src_padded_nodes_neighbor_ids, ndarray, shape (batch_size, src_max_seq_length)
        # src_padded_nodes_edge_ids, ndarray, shape (batch_size, src_max_seq_length)
        # src_padded_nodes_neighbor_times, ndarray, shape (batch_size, src_max_seq_length)
        (
            src_padded_nodes_neighbor_ids,
            src_padded_nodes_edge_ids,
            src_padded_nodes_neighbor_times,
            src_padded_nodes_neighbor_sign,
        ) = self.pad_sequences(
            node_ids=src_node_ids,
            node_interact_times=node_interact_times,
            nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
            nodes_edge_ids_list=src_nodes_edge_ids_list,
            nodes_neighbor_times_list=src_nodes_neighbor_times_list,
            node_interact_sign=node_interact_sign,
            nodes_neighbor_sign_list=src_nodes_neighbor_sign_list,
            patch_size=self.patch_size,
            max_input_sequence_length=self.max_input_sequence_length,
        )

        # dst_padded_nodes_neighbor_ids, ndarray, shape (batch_size, dst_max_seq_length)
        # dst_padded_nodes_edge_ids, ndarray, shape (batch_size, dst_max_seq_length)
        # dst_padded_nodes_neighbor_times, ndarray, shape (batch_size, dst_max_seq_length)
        (
            dst_padded_nodes_neighbor_ids,
            dst_padded_nodes_edge_ids,
            dst_padded_nodes_neighbor_times,
            dst_padded_nodes_neighbor_sign,
        ) = self.pad_sequences(
            node_ids=dst_node_ids,
            node_interact_times=node_interact_times,
            nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
            nodes_edge_ids_list=dst_nodes_edge_ids_list,
            nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
            node_interact_sign=node_interact_sign,
            nodes_neighbor_sign_list=dst_nodes_neighbor_sign_list,
            patch_size=self.patch_size,
            max_input_sequence_length=self.max_input_sequence_length,
        )

        # src_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
        # dst_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
        (
            src_padded_nodes_neighbor_co_occurrence_features,
            dst_padded_nodes_neighbor_co_occurrence_features,
        ) = self.neighbor_co_occurrence_encoder.forward(
            src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
            dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
            node_interact_sign=node_interact_sign,
            src_padded_nodes_neighbor_sign=src_padded_nodes_neighbor_sign,
            dst_padded_nodes_neighbor_sign=dst_padded_nodes_neighbor_sign,
        )

        # get the features of the sequence of source and destination nodes
        # src_padded_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, src_max_seq_length, node_feat_dim)
        # src_padded_nodes_edge_raw_features, Tensor, shape (batch_size, src_max_seq_length, edge_feat_dim)
        # src_padded_nodes_neighbor_time_features, Tensor, shape (batch_size, src_max_seq_length, time_feat_dim)
        (
            src_padded_nodes_neighbor_node_raw_features,
            src_padded_nodes_edge_raw_features,
            src_padded_nodes_neighbor_time_features,
        ) = self.get_features(
            node_interact_times=node_interact_times,
            padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
            padded_nodes_edge_ids=src_padded_nodes_edge_ids,
            padded_nodes_neighbor_times=src_padded_nodes_neighbor_times,
            time_encoder=self.time_encoder,
        )

        # dst_padded_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, dst_max_seq_length, node_feat_dim)
        # dst_padded_nodes_edge_raw_features, Tensor, shape (batch_size, dst_max_seq_length, edge_feat_dim)
        # dst_padded_nodes_neighbor_time_features, Tensor, shape (batch_size, dst_max_seq_length, time_feat_dim)
        (
            dst_padded_nodes_neighbor_node_raw_features,
            dst_padded_nodes_edge_raw_features,
            dst_padded_nodes_neighbor_time_features,
        ) = self.get_features(
            node_interact_times=node_interact_times,
            padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
            padded_nodes_edge_ids=dst_padded_nodes_edge_ids,
            padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times,
            time_encoder=self.time_encoder,
        )

        # get the patches for source and destination nodes
        # src_patches_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, src_num_patches, patch_size * node_feat_dim)
        # src_patches_nodes_edge_raw_features, Tensor, shape (batch_size, src_num_patches, patch_size * edge_feat_dim)
        # src_patches_nodes_neighbor_time_features, Tensor, shape (batch_size, src_num_patches, patch_size * time_feat_dim)
        (
            src_patches_nodes_neighbor_node_raw_features,
            src_patches_nodes_edge_raw_features,
            src_patches_nodes_neighbor_time_features,
            src_patches_nodes_neighbor_co_occurrence_features,
        ) = self.get_patches(
            padded_nodes_neighbor_node_raw_features=src_padded_nodes_neighbor_node_raw_features,
            padded_nodes_edge_raw_features=src_padded_nodes_edge_raw_features,
            padded_nodes_neighbor_time_features=src_padded_nodes_neighbor_time_features,
            padded_nodes_neighbor_co_occurrence_features=src_padded_nodes_neighbor_co_occurrence_features,
            patch_size=self.patch_size,
        )

        # dst_patches_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, dst_num_patches, patch_size * node_feat_dim)
        # dst_patches_nodes_edge_raw_features, Tensor, shape (batch_size, dst_num_patches, patch_size * edge_feat_dim)
        # dst_patches_nodes_neighbor_time_features, Tensor, shape (batch_size, dst_num_patches, patch_size * time_feat_dim)
        (
            dst_patches_nodes_neighbor_node_raw_features,
            dst_patches_nodes_edge_raw_features,
            dst_patches_nodes_neighbor_time_features,
            dst_patches_nodes_neighbor_co_occurrence_features,
        ) = self.get_patches(
            padded_nodes_neighbor_node_raw_features=dst_padded_nodes_neighbor_node_raw_features,
            padded_nodes_edge_raw_features=dst_padded_nodes_edge_raw_features,
            padded_nodes_neighbor_time_features=dst_padded_nodes_neighbor_time_features,
            padded_nodes_neighbor_co_occurrence_features=dst_padded_nodes_neighbor_co_occurrence_features,
            patch_size=self.patch_size,
        )

        # align the patch encoding dimension
        # Tensor, shape (batch_size, src_num_patches, channel_embedding_dim)
        src_patches_nodes_neighbor_node_raw_features = self.projection_layer["node"](
            src_patches_nodes_neighbor_node_raw_features
        )
        src_patches_nodes_edge_raw_features = self.projection_layer["edge"](
            src_patches_nodes_edge_raw_features
        )
        src_patches_nodes_neighbor_time_features = self.projection_layer["time"](
            src_patches_nodes_neighbor_time_features
        )
        src_patches_nodes_neighbor_co_occurrence_features = self.projection_layer[
            "neighbor_co_occurrence"
        ](src_patches_nodes_neighbor_co_occurrence_features)

        # Tensor, shape (batch_size, dst_num_patches, channel_embedding_dim)
        dst_patches_nodes_neighbor_node_raw_features = self.projection_layer["node"](
            dst_patches_nodes_neighbor_node_raw_features
        )
        dst_patches_nodes_edge_raw_features = self.projection_layer["edge"](
            dst_patches_nodes_edge_raw_features
        )
        dst_patches_nodes_neighbor_time_features = self.projection_layer["time"](
            dst_patches_nodes_neighbor_time_features
        )
        dst_patches_nodes_neighbor_co_occurrence_features = self.projection_layer[
            "neighbor_co_occurrence"
        ](dst_patches_nodes_neighbor_co_occurrence_features)

        batch_size = len(src_patches_nodes_neighbor_node_raw_features)
        src_num_patches = src_patches_nodes_neighbor_node_raw_features.shape[1]
        dst_num_patches = dst_patches_nodes_neighbor_node_raw_features.shape[1]

        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, channel_embedding_dim)
        patches_nodes_neighbor_node_raw_features = torch.cat(
            [
                src_patches_nodes_neighbor_node_raw_features,
                dst_patches_nodes_neighbor_node_raw_features,
            ],
            dim=1,
        )
        patches_nodes_edge_raw_features = torch.cat(
            [src_patches_nodes_edge_raw_features, dst_patches_nodes_edge_raw_features],
            dim=1,
        )
        patches_nodes_neighbor_time_features = torch.cat(
            [
                src_patches_nodes_neighbor_time_features,
                dst_patches_nodes_neighbor_time_features,
            ],
            dim=1,
        )
        patches_nodes_neighbor_co_occurrence_features = torch.cat(
            [
                src_patches_nodes_neighbor_co_occurrence_features,
                dst_patches_nodes_neighbor_co_occurrence_features,
            ],
            dim=1,
        )

        patches_data = [
            patches_nodes_neighbor_node_raw_features,
            patches_nodes_edge_raw_features,
            patches_nodes_neighbor_time_features,
            patches_nodes_neighbor_co_occurrence_features,
        ]
        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels, channel_embedding_dim)
        patches_data = torch.stack(patches_data, dim=2)
        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels * channel_embedding_dim)
        patches_data = patches_data.reshape(
            batch_size,
            src_num_patches + dst_num_patches,
            self.num_channels * self.channel_embedding_dim,
        )

        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels * channel_embedding_dim)
        for transformer in self.transformers:
            patches_data = transformer(patches_data)

        # src_patches_data, Tensor, shape (batch_size, src_num_patches, num_channels * channel_embedding_dim)
        src_patches_data = patches_data[:, :src_num_patches, :]
        # dst_patches_data, Tensor, shape (batch_size, dst_num_patches, num_channels * channel_embedding_dim)
        dst_patches_data = patches_data[
            :, src_num_patches : src_num_patches + dst_num_patches, :
        ]
        # src_patches_data, Tensor, shape (batch_size, num_channels * channel_embedding_dim)
        src_patches_data = torch.mean(src_patches_data, dim=1)
        # dst_patches_data, Tensor, shape (batch_size, num_channels * channel_embedding_dim)
        dst_patches_data = torch.mean(dst_patches_data, dim=1)

        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings = self.output_layer(src_patches_data)
        # Tensor, shape (batch_size, node_feat_dim)
        dst_node_embeddings = self.output_layer(dst_patches_data)

        return src_node_embeddings, dst_node_embeddings

    def pad_sequences(
        self,
        node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        nodes_neighbor_ids_list: list,
        nodes_edge_ids_list: list,
        nodes_neighbor_times_list: list,
        node_interact_sign: np.ndarray,
        nodes_neighbor_sign_list: list,
        patch_size: int = 1,
        max_input_sequence_length: int = 256,
    ):
        """
        pad the sequences for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param nodes_neighbor_ids_list: list of ndarrays, each ndarray contains neighbor ids for nodes in node_ids
        :param nodes_edge_ids_list: list of ndarrays, each ndarray contains edge ids for nodes in node_ids
        :param nodes_neighbor_times_list: list of ndarrays, each ndarray contains neighbor interaction timestamp for nodes in node_ids
        :param patch_size: int, patch size
        :param max_input_sequence_length: int, maximal number of neighbors for each node
        :return:
        """
        assert (
            max_input_sequence_length - 1 > 0
        ), "Maximal number of neighbors for each node should be greater than 1!"
        max_seq_length = 0
        # first cut the sequence of nodes whose number of neighbors is more than max_input_sequence_length - 1 (we need to include the target node in the sequence)
        for idx in range(len(nodes_neighbor_ids_list)):
            assert (
                len(nodes_neighbor_ids_list[idx])
                == len(nodes_edge_ids_list[idx])
                == len(nodes_neighbor_times_list[idx])
                == len(nodes_neighbor_sign_list[idx])
            )
            # 截断过长
            if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
                # cut the sequence by taking the most recent max_input_sequence_length interactions
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][
                    -(max_input_sequence_length - 1) :
                ]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][
                    -(max_input_sequence_length - 1) :
                ]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][
                    -(max_input_sequence_length - 1) :
                ]
                nodes_neighbor_sign_list[idx] = nodes_neighbor_sign_list[idx][
                    -(max_input_sequence_length - 1) :
                ]
            if len(nodes_neighbor_ids_list[idx]) > max_seq_length:
                max_seq_length = len(nodes_neighbor_ids_list[idx])

        # include the target node itself
        max_seq_length += 1
        if max_seq_length % patch_size != 0:
            max_seq_length += patch_size - max_seq_length % patch_size
        assert max_seq_length % patch_size == 0

        # 对齐序列,全部填充0
        # pad the sequences
        # three ndarrays with shape (batch_size, max_seq_length)
        padded_nodes_neighbor_ids = np.zeros((len(node_ids), max_seq_length)).astype(
            np.longlong
        )
        padded_nodes_edge_ids = np.zeros((len(node_ids), max_seq_length)).astype(
            np.longlong
        )
        padded_nodes_neighbor_times = np.zeros((len(node_ids), max_seq_length)).astype(
            np.float32
        )
        padded_nodes_neighbor_sign = np.zeros((len(node_ids), max_seq_length)).astype(
            np.int8
        )

        for idx in range(len(node_ids)):
            # 第一个元素填写当前交互的信息
            padded_nodes_neighbor_ids[idx, 0] = node_ids[idx]
            padded_nodes_edge_ids[idx, 0] = 0
            padded_nodes_neighbor_times[idx, 0] = node_interact_times[idx]
            padded_nodes_neighbor_sign[idx, 0] = node_interact_sign[idx]

            # 余下元素填写剩下（低到高对齐，idx越小约靠近，idx越大越远）
            if len(nodes_neighbor_ids_list[idx]) > 0:
                padded_nodes_neighbor_ids[
                    idx, 1 : len(nodes_neighbor_ids_list[idx]) + 1
                ] = nodes_neighbor_ids_list[idx]

                padded_nodes_edge_ids[idx, 1 : len(nodes_edge_ids_list[idx]) + 1] = (
                    nodes_edge_ids_list[idx]
                )
                padded_nodes_neighbor_times[
                    idx, 1 : len(nodes_neighbor_times_list[idx]) + 1
                ] = nodes_neighbor_times_list[idx]

                padded_nodes_neighbor_sign[
                    idx, 1 : len(nodes_neighbor_sign_list[idx]) + 1
                ] = nodes_neighbor_sign_list[idx]

        # three ndarrays with shape (batch_size, max_seq_length)
        return (
            padded_nodes_neighbor_ids,
            padded_nodes_edge_ids,
            padded_nodes_neighbor_times,
            padded_nodes_neighbor_sign,
        )

    def get_features(
        self,
        node_interact_times: np.ndarray,
        padded_nodes_neighbor_ids: np.ndarray,
        padded_nodes_edge_ids: np.ndarray,
        padded_nodes_neighbor_times: np.ndarray,
        time_encoder: TimeEncoder,
    ):
        """
        get node, edge and time features
        :param node_interact_times: ndarray, shape (batch_size, )
        :param padded_nodes_neighbor_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_edge_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_neighbor_times: ndarray, shape (batch_size, max_seq_length)
        :param time_encoder: TimeEncoder, time encoder
        :return:
        """
        # Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        padded_nodes_neighbor_node_raw_features = self.node_raw_features[
            torch.from_numpy(padded_nodes_neighbor_ids)
        ]
        # Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        padded_nodes_edge_raw_features = self.edge_raw_features[
            torch.from_numpy(padded_nodes_edge_ids)
        ]
        # Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        padded_nodes_neighbor_time_features = time_encoder(
            timestamps=torch.from_numpy(
                node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times
            )
            .float()
            .to(self.device)
        )

        # ndarray, set the time features to all zeros for the padded timestamp
        padded_nodes_neighbor_time_features[
            torch.from_numpy(padded_nodes_neighbor_ids == 0)
        ] = 0.0

        return (
            padded_nodes_neighbor_node_raw_features,
            padded_nodes_edge_raw_features,
            padded_nodes_neighbor_time_features,
        )

    def get_patches(
        self,
        padded_nodes_neighbor_node_raw_features: torch.Tensor,
        padded_nodes_edge_raw_features: torch.Tensor,
        padded_nodes_neighbor_time_features: torch.Tensor,
        padded_nodes_neighbor_co_occurrence_features: torch.Tensor = None,
        patch_size: int = 1,
    ):
        """
        get the sequence of patches for nodes
        :param padded_nodes_neighbor_node_raw_features: Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        :param padded_nodes_edge_raw_features: Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        :param padded_nodes_neighbor_time_features: Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        :param padded_nodes_neighbor_co_occurrence_features: Tensor, shape (batch_size, max_seq_length, neighbor_co_occurrence_feat_dim)
        :param patch_size: int, patch size
        :return:
        """
        assert padded_nodes_neighbor_node_raw_features.shape[1] % patch_size == 0
        num_patches = padded_nodes_neighbor_node_raw_features.shape[1] // patch_size

        # list of Tensors with shape (num_patches, ), each Tensor with shape (batch_size, patch_size, node_feat_dim)
        (
            patches_nodes_neighbor_node_raw_features,
            patches_nodes_edge_raw_features,
            patches_nodes_neighbor_time_features,
            patches_nodes_neighbor_co_occurrence_features,
        ) = ([], [], [], [])

        for patch_id in range(num_patches):
            start_idx = patch_id * patch_size
            end_idx = patch_id * patch_size + patch_size
            patches_nodes_neighbor_node_raw_features.append(
                padded_nodes_neighbor_node_raw_features[:, start_idx:end_idx, :]
            )
            patches_nodes_edge_raw_features.append(
                padded_nodes_edge_raw_features[:, start_idx:end_idx, :]
            )
            patches_nodes_neighbor_time_features.append(
                padded_nodes_neighbor_time_features[:, start_idx:end_idx, :]
            )
            patches_nodes_neighbor_co_occurrence_features.append(
                padded_nodes_neighbor_co_occurrence_features[:, start_idx:end_idx, :]
            )

        batch_size = len(padded_nodes_neighbor_node_raw_features)
        # Tensor, shape (batch_size, num_patches, patch_size * node_feat_dim)
        patches_nodes_neighbor_node_raw_features = torch.stack(
            patches_nodes_neighbor_node_raw_features, dim=1
        ).reshape(batch_size, num_patches, patch_size * self.node_feat_dim)
        # Tensor, shape (batch_size, num_patches, patch_size * edge_feat_dim)
        patches_nodes_edge_raw_features = torch.stack(
            patches_nodes_edge_raw_features, dim=1
        ).reshape(batch_size, num_patches, patch_size * self.edge_feat_dim)
        # Tensor, shape (batch_size, num_patches, patch_size * time_feat_dim)
        patches_nodes_neighbor_time_features = torch.stack(
            patches_nodes_neighbor_time_features, dim=1
        ).reshape(batch_size, num_patches, patch_size * self.time_feat_dim)

        patches_nodes_neighbor_co_occurrence_features = torch.stack(
            patches_nodes_neighbor_co_occurrence_features, dim=1
        ).reshape(
            batch_size, num_patches, patch_size * self.neighbor_co_occurrence_feat_dim
        )

        return (
            patches_nodes_neighbor_node_raw_features,
            patches_nodes_edge_raw_features,
            patches_nodes_neighbor_time_features,
            patches_nodes_neighbor_co_occurrence_features,
        )

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in [
            "uniform",
            "time_interval_aware",
        ]:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


class NeighborCooccurrenceEncoder(nn.Module):

    def __init__(self, neighbor_co_occurrence_feat_dim: int, device: str = "cpu"):
        """
        Neighbor co-occurrence encoder.
        :param neighbor_co_occurrence_feat_dim: int, dimension of neighbor co-occurrence features (encodings)
        :param device: str, device
        """
        super(NeighborCooccurrenceEncoder, self).__init__()
        self.neighbor_co_occurrence_feat_dim = neighbor_co_occurrence_feat_dim
        self.device = device

        self.neighbor_co_occurrence_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.neighbor_co_occurrence_feat_dim),
            nn.ReLU(),
            nn.Linear(
                in_features=self.neighbor_co_occurrence_feat_dim,
                out_features=self.neighbor_co_occurrence_feat_dim,
            ),
        )

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

    def to_float_torch(self, arr: np.ndarray, apply: Union[Callable , None] = None):
        tensor = torch.from_numpy(arr)

        if apply is not None:
            tensor = tensor.apply_(apply)

        tensor = tensor.float().to(self.device)

        return tensor

    def count_nodes_appearances(
        self,
        src_padded_nodes_neighbor_ids: np.ndarray,
        dst_padded_nodes_neighbor_ids: np.ndarray,
        node_interact_sign: np.ndarray,
        src_padded_nodes_neighbor_sign: np.ndarray,
        dst_padded_nodes_neighbor_sign: np.ndarray,
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
            node_sign,
            src_padded_node_neighbor_sign,
            dst_padded_node_neighbor_sign,
        ) in zip(
            src_padded_nodes_neighbor_ids,
            dst_padded_nodes_neighbor_ids,
            node_interact_sign,
            src_padded_nodes_neighbor_sign,
            dst_padded_nodes_neighbor_sign,
        ):

            ctrl_col = None
            if node_sign == 1:
                ctrl_col = 0
            elif node_sign == -1:
                ctrl_col = 1

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

            # src历史邻居符号编码
            (
                src_padded_pos_node_neighbor_signed_counts,
                src_padded_neg_node_neighbor_signed_counts,
                src_sign_mapping_dict,
            ) = self.sign_neighbor_count(
                src_padded_node_neighbor_ids, src_padded_node_neighbor_sign
            )

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

            # dst 历史邻居符号编码
            (
                dst_padded_pos_node_neighbor_signed_counts,
                dst_padded_neg_node_neighbor_signed_counts,
                dst_sign_mapping_dict,
            ) = self.sign_neighbor_count(
                dst_padded_node_neighbor_ids, dst_padded_node_neighbor_sign
            )

            # 更新src_padded_pos_node_neighbor_signed_counts， 基于交互的符号，将dst的计数也拿来
            if ctrl_col is None:
                src_pos_adds = np.zeros(len(src_padded_node_neighbor_ids))
            else:
                src_pos_adds = np.array(
                    [
                        dst_sign_mapping_dict.get(i, (0, 0))[ctrl_col]
                        for i in src_padded_node_neighbor_ids
                    ]
                )
            # src_padded_pos_node_neighbor_signed_counts += src_pos_adds
            src_padded_pos_node_neighbor_signed_counts_in_src = self.to_float_torch(
                src_padded_pos_node_neighbor_signed_counts
            )

            # 更新src_padded_neg_node_neighbor_signed_counts， 基于交互编码的符号，将dst的计数也拿过来
            if ctrl_col is None:
                src_neg_adds = np.zeros(len(src_padded_node_neighbor_ids))
            else:
                src_neg_adds = np.array(
                    [
                        dst_sign_mapping_dict.get(i, (0, 0))[1 - ctrl_col]
                        for i in src_padded_node_neighbor_ids
                    ]
                )
            # src_padded_neg_node_neighbor_signed_counts += src_neg_adds
            src_padded_neg_node_neighbor_signed_counts_in_src = self.to_float_torch(
                src_padded_neg_node_neighbor_signed_counts
            )

            # 更新dst_padded_pos_node_neighbor_signed_counts基于交互的符号，将src的计数拿过来
            if ctrl_col is None:
                dst_pos_adds = np.zeros(len(dst_padded_node_neighbor_ids))
            else:
                dst_pos_adds = np.array(
                    [
                        src_sign_mapping_dict.get(i, (0, 0))[ctrl_col]
                        for i in dst_padded_node_neighbor_ids
                    ]
                )
            # dst_padded_pos_node_neighbor_signed_counts += dst_pos_adds
            dst_padded_pos_node_neighbor_signed_counts_in_dst = self.to_float_torch(
                dst_padded_pos_node_neighbor_signed_counts
            )

            # 更新dst_padded_pos_node_neighbor_signed_counts基于交互的符号，将src的计数拿过来
            if ctrl_col is None:
                dst_neg_adds = np.zeros(len(dst_padded_node_neighbor_ids))
            else:
                dst_neg_adds = np.array(
                    [
                        src_sign_mapping_dict.get(i, (0, 0))[1 - ctrl_col]
                        for i in dst_padded_node_neighbor_ids
                    ]
                )
            # dst_padded_neg_node_neighbor_signed_counts += dst_neg_adds
            dst_padded_neg_node_neighbor_signed_counts_in_dst = self.to_float_torch(
                dst_padded_neg_node_neighbor_signed_counts
            )

            # we need to use copy() to avoid the modification of src_padded_node_neighbor_ids
            # Tensor, shape (src_max_seq_length, )
            # src 节点的历史邻居在dst里面的出现
            src_padded_node_neighbor_counts_in_dst = self.to_float_torch(
                src_padded_node_neighbor_ids.copy(),
                lambda neighbor_id: dst_mapping_dict.get(neighbor_id, 0.0),
            )

            src_padded_pos_node_neighbor_signed_counts_in_dst = self.to_float_torch(
                src_padded_node_neighbor_ids.copy(),
                lambda nid: (
                    dst_sign_mapping_dict.get(nid, (0.0, 0.0))[0]
                    # + 0
                    # if ctrl_col is None
                    # else src_sign_mapping_dict.get(nid, (0.0, 0.0))[ctrl_col]
                ),
            )

            src_padded_neg_node_neighbor_signed_counts_in_dst = self.to_float_torch(
                src_padded_node_neighbor_ids.copy(),
                lambda nid: (
                    dst_sign_mapping_dict.get(nid, (0.0, 0.0))[1]
                    # + 0
                    # if ctrl_col is None
                    # else src_sign_mapping_dict.get(nid, (0.0, 0.0))[1- ctrl_col]
                ),
            )

            # Tensor, shape (src_max_seq_length, 2)
            src_padded_nodes_appearances.append(
                torch.stack(
                    [
                        src_padded_node_neighbor_counts_in_src,
                        # src positive node neighbor counts
                        src_padded_pos_node_neighbor_signed_counts_in_src,
                        # src negative node neighbor counts
                        src_padded_neg_node_neighbor_signed_counts_in_src,
                        src_padded_node_neighbor_counts_in_dst,
                        # src pos node neg counts in dst
                        src_padded_pos_node_neighbor_signed_counts_in_dst,
                        # src pos node neg counts in dst
                        src_padded_neg_node_neighbor_signed_counts_in_dst,
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

            dst_padded_pos_node_neighbor_signed_counts_in_src = self.to_float_torch(
                dst_padded_node_neighbor_ids.copy(),
                lambda nid: src_sign_mapping_dict.get(nid, (0.0, 0.0))[0],
                #  + 0
                # if ctrl_col is None
            )

            dst_padded_neg_node_neighbor_signed_counts_in_src = self.to_float_torch(
                dst_padded_node_neighbor_ids.copy(),
                lambda nid: (
                    src_sign_mapping_dict.get(nid, (0.0, 0.0))[1]
                    # + 0
                    # if ctrl_col is None
                    # else dst_sign_mapping_dict.get(nid, (0.0, 0.0))[1 - ctrl_col]
                ),
            )

            # Tensor, shape (dst_max_seq_length, 2)
            dst_padded_nodes_appearances.append(
                torch.stack(
                    [
                        dst_padded_node_neighbor_counts_in_src,
                        dst_padded_pos_node_neighbor_signed_counts_in_dst,
                        dst_padded_neg_node_neighbor_signed_counts_in_dst,
                        dst_padded_node_neighbor_counts_in_dst,
                        dst_padded_pos_node_neighbor_signed_counts_in_src,
                        dst_padded_neg_node_neighbor_signed_counts_in_src,
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

    def forward(
        self,
        src_padded_nodes_neighbor_ids: np.ndarray,
        dst_padded_nodes_neighbor_ids: np.ndarray,
        node_interact_sign: np.ndarray,
        src_padded_nodes_neighbor_sign: np.ndarray,
        dst_padded_nodes_neighbor_sign: np.ndarray,
    ):
        """
        compute the neighbor co-occurrence features of nodes in src_padded_nodes_neighbor_ids and dst_padded_nodes_neighbor_ids
        :param src_padded_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_padded_nodes_neighbor_ids:: ndarray, shape (batch_size, dst_max_seq_length)
        :return:
        """
        # src_padded_nodes_appearances, Tensor, shape (batch_size, src_max_seq_length, 2)
        # dst_padded_nodes_appearances, Tensor, shape (batch_size, dst_max_seq_length, 2)
        src_padded_nodes_appearances, dst_padded_nodes_appearances = (
            self.count_nodes_appearances(
                src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                node_interact_sign=node_interact_sign,
                src_padded_nodes_neighbor_sign=src_padded_nodes_neighbor_sign,
                dst_padded_nodes_neighbor_sign=dst_padded_nodes_neighbor_sign,
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

        # src_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
        # dst_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
        return (
            src_padded_nodes_neighbor_co_occurrence_features,
            dst_padded_nodes_neighbor_co_occurrence_features,
        )


class TransformerEncoder(nn.Module):

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Transformer encoder.
        :param attention_dim: int, dimension of the attention vector
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(TransformerEncoder, self).__init__()
        # use the MultiheadAttention implemented by PyTorch
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
        """
        encode the inputs by Transformer encoder
        :param inputs: Tensor, shape (batch_size, num_patches, self.attention_dim)
        :return:
        """
        # note that the MultiheadAttention module accept input data with shape (seq_length, batch_size, input_dim), so we need to transpose the input
        # Tensor, shape (num_patches, batch_size, self.attention_dim)
        transposed_inputs = inputs.transpose(0, 1)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        transposed_inputs = self.norm_layers[0](transposed_inputs)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        hidden_states = self.multi_head_attention(
            query=transposed_inputs, key=transposed_inputs, value=transposed_inputs
        )[0].transpose(0, 1)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        outputs = inputs + self.dropout(hidden_states)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        hidden_states = self.linear_layers[1](
            self.dropout(F.gelu(self.linear_layers[0](self.norm_layers[1](outputs))))
        )
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        outputs = outputs + self.dropout(hidden_states)
        return outputs
