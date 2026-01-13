"""带方向的历史交互邻居采样器"""

from enum import Enum
from typing import Dict, List, Optional, OrderedDict, Tuple
import numpy as np
import torch
from tqdm import tqdm

from utils.DataLoader import Data


class NeighborType(Enum):
    # 邻居为有向边的源节点，自身为目标节点
    IncomeNeighbor = 1
    # 邻居为有向边的目标节点，自身为源节点
    OutcomeNeighbor = 2


class Neighbor:
    def __init__(
        self, dst: int, edge_id: int, timestamp: float, sign: int, ty: NeighborType
    ):
        self.id: int = dst
        self.edge_id: int = edge_id
        self.timestamp: float = timestamp
        self.sign: int = sign
        self.ty: NeighborType = ty

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"id={self.id}, "
            f"timestamp={self.timestamp}, "
            f'sign=< {"+" if self.sign == 1 else "-"} >, '
            f"type={self.ty.name}"
        )


class NeighborGroup:
    def __init__(self):
        self.ids: List[np.ndarray] = []
        self.edges_ids: List[np.ndarray] = []
        self.times: List[np.ndarray] = []
        self.signs: List[np.ndarray] = []

    def push_neighbors(
        self,
        neighbors: List[Neighbor],
        *,
        ordered_seq: bool = False,
        directed_filter: Optional[NeighborType] = None,
    ):
        filtered_neighbors = filter(
            lambda x: True if directed_filter is None else (x.ty == directed_filter),
            neighbors,
        )
        if ordered_seq:
            sorted_neighbors = list(filtered_neighbors)
        else:
            sorted_neighbors = sorted(filtered_neighbors, key=lambda x: x.timestamp)

        self.ids.append(np.array([x.id for x in sorted_neighbors]))
        self.edges_ids.append(np.array([x.edge_id for x in sorted_neighbors]))
        self.times.append(np.array([x.timestamp for x in sorted_neighbors]))
        self.signs.append(np.array([x.sign for x in sorted_neighbors]))

    def __len__(self):
        return len(self.ids)


def common_positions_np(a, b):
    """
    a, b : 1-D numpy array
    return : dict{value: (idx_a, idx_b)}
    """
    # 对 a 做唯一化 + 逆索引
    uniq_a, inv_a = np.unique(a, return_inverse=True)
    # 把相同 value 的下标按 value 分组
    pos_a = {v: np.where(inv_a == i)[0] for i, v in enumerate(uniq_a)}

    # 对 b 同理
    uniq_b, inv_b = np.unique(b, return_inverse=True)
    pos_b = {v: np.where(inv_b == i)[0] for i, v in enumerate(uniq_b)}

    # 交集 & 组装
    common_vals = np.intersect1d(uniq_a, uniq_b, assume_unique=True)
    return {int(v): (pos_a[v], pos_b[v]) for v in common_vals}


class DirectedNeighborSampler:

    def __init__(
        self,
        adj_list: Dict[int, List[Neighbor]],
        sample_neighbor_strategy: str = "uniform",
        time_scaling_factor: float = 0.0,
        seed: int = None,
        common_neighbor_look_forward: int = 5,
    ):
        """
        Neighbor sampler.
        :param adj_list: list, list of list, where each element is a list of triple tuple (node_id, edge_id, timestamp)
        :param sample_neighbor_strategy: str, how to sample historical neighbors, 'uniform', 'recent', or 'time_interval_aware'
        :param time_scaling_factor: float, a hyper-parameter that controls the sampling preference with time interval,
        a large time_scaling_factor tends to sample more on recent links, this parameter works when sample_neighbor_strategy == 'time_interval_aware'
        :param seed: int, random seed
        """
        self.sample_neighbor_strategy = sample_neighbor_strategy
        self.seed = seed
        self.common_neighbors_look_forward = common_neighbor_look_forward

        # list of each node's neighbor ids, edge ids and interaction times, which are sorted by interaction times
        # 无符号时使用的邻居，不考虑邻居符号信息
        self.undirected_nodes_neighbor = NeighborGroup()

        # 有符号邻居，分别为入度邻居和出度邻居
        # 入度的节点历史交互邻居
        self.directed_nodes_in_neighbor = NeighborGroup()
        # 出度的历史交互邻居
        self.directed_nodes_out_neighbor = NeighborGroup()

        if self.sample_neighbor_strategy == "time_interval_aware":
            self.nodes_neighbor_sampled_probabilities = []
            self.time_scaling_factor = time_scaling_factor

        # 检查是否节点是有序的
        n = len(adj_list)
        assert set(adj_list) == set(
            range(0, n)
        ), f"给定的邻接表的节点编号不是严格递增的连续序列"
        # the list at the first position in adj_list is empty, hence, sorted() will return an empty list for the first position
        # its corresponding value in self.nodes_neighbor_ids, self.nodes_edge_ids, self.nodes_neighbor_times will also be empty with length 0
        previous_node_idx = -1
        for node_idx, per_node_neighbors in tqdm(
            sorted(adj_list.items()), desc="loading node neighbors"
        ):
            # per_node_neighbors is a list of tuples (neighbor_id, edge_id, timestamp)
            # sort the list based on timestamps, sorted() function is stable
            # Note that sort the list based on edge id is also correct, as the original data file ensures the interactions are chronological
            sorted_neighbors = sorted(per_node_neighbors, key=lambda x: x.timestamp)
            self.undirected_nodes_neighbor.push_neighbors(
                per_node_neighbors, ordered_seq=True
            )
            self.directed_nodes_in_neighbor.push_neighbors(
                per_node_neighbors,
                directed_filter=NeighborType.IncomeNeighbor,
                ordered_seq=True,
            )
            self.directed_nodes_out_neighbor.push_neighbors(
                per_node_neighbors,
                directed_filter=NeighborType.OutcomeNeighbor,
                ordered_seq=True,
            )
            previous_node_idx = node_idx

            # additional for time interval aware sampling strategy (proposed in CAWN paper)
            if self.sample_neighbor_strategy == "time_interval_aware":
                self.nodes_neighbor_sampled_probabilities.append(
                    self.compute_sampled_probabilities(
                        np.array([x.timestamp for x in sorted_neighbors])
                    )
                )
        print(f"Max node idx: {previous_node_idx}")

        assert (
            previous_node_idx + 1
            == len(self.undirected_nodes_neighbor)
            == len(self.directed_nodes_in_neighbor)
            == len(self.directed_nodes_out_neighbor)
        ), ""
        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def compute_sampled_probabilities(self, node_neighbor_times: np.ndarray):
        """
        compute the sampled probabilities of historical neighbors based on their interaction times
        :param node_neighbor_times: ndarray, shape (num_historical_neighbors, )
        :return:
        """
        if len(node_neighbor_times) == 0:
            return np.array([])
        # compute the time delta with regard to the last time in node_neighbor_times
        node_neighbor_times = node_neighbor_times - np.max(node_neighbor_times)
        # compute the normalized sampled probabilities of historical neighbors
        exp_node_neighbor_times = np.exp(self.time_scaling_factor * node_neighbor_times)
        sampled_probabilities = exp_node_neighbor_times / np.cumsum(
            exp_node_neighbor_times
        )
        # note that the first few values in exp_node_neighbor_times may be all zero, which make the corresponding values in sampled_probabilities
        # become nan (divided by zero), so we replace the nan by a very large negative number -1e10 to denote the sampled probabilities
        sampled_probabilities[np.isnan(sampled_probabilities)] = -1e10
        return sampled_probabilities

    def nodes_neighbors_selector(self, neighbor_ty: Optional[NeighborType] = None):
        # 基于neighbor_ty 选择使用哪个邻居
        # 未提供，无向图
        if neighbor_ty is None:
            nodes_neighbors = self.undirected_nodes_neighbor
        # 入度邻居
        elif neighbor_ty == NeighborType.IncomeNeighbor:
            nodes_neighbors = self.directed_nodes_in_neighbor
        elif neighbor_ty == NeighborType.OutcomeNeighbor:
            nodes_neighbors = self.directed_nodes_out_neighbor
        else:
            raise ValueError(f"Unknown request Neighbor type: [{neighbor_ty}]")

        return nodes_neighbors

    def find_neighbors_before(
        self,
        node_id: int,
        interact_time: float,
        *,
        neighbor_ty: Optional[NeighborType] = None,
        return_sampled_probabilities: bool = False,
    ):
        """
        extracts all the interactions happening before interact_time (less than interact_time) for node_id in the overall interaction graph
        the returned interactions are sorted by time.
        :param node_id: int, node id
        :param interact_time: float, interaction time
        :param return_sampled_probabilities: boolean, whether return the sampled probabilities of neighbors
        :return: neighbors, edge_ids, timestamps,sign and sampled_probabilities (if return_sampled_probabilities is True) with shape (historical_nodes_num, )
        """
        nodes_neighbors = self.nodes_neighbors_selector(neighbor_ty)
        assert node_id <= len(
            nodes_neighbors
        ), f"节点ID[{node_id}]大于历史交互邻居记录节点ID[{len(nodes_neighbor_ids)-1}]"

        # return index i, which satisfies list[i - 1] < v <= list[i]
        # return 0 for the first position in self.nodes_neighbor_times since the value at the first position is empty
        try:
            i = np.searchsorted(nodes_neighbors.times[node_id], interact_time)
        except IndexError:
            print(
                f"Detect Index Error, request idx: {node_id}, max_list len: {len(nodes_neighbors.times)}"
            )
            raise
        (
            nodes_neighbor_ids,
            nodes_edge_ids,
            nodes_neighbor_times,
            nodes_neighbor_signs,
        ) = (
            nodes_neighbors.ids,
            nodes_neighbors.edges_ids,
            nodes_neighbors.times,
            nodes_neighbors.signs,
        )

        if return_sampled_probabilities:
            return (
                nodes_neighbor_ids[node_id][:i],
                nodes_edge_ids[node_id][:i],
                nodes_neighbor_times[node_id][:i],
                nodes_neighbor_signs[node_id][:i],
                self.nodes_neighbor_sampled_probabilities[node_id][:i],
            )
        else:
            return (
                nodes_neighbor_ids[node_id][:i],
                nodes_edge_ids[node_id][:i],
                nodes_neighbor_times[node_id][:i],
                nodes_neighbor_signs[node_id][:i],
                None,
            )

    def get_all_first_hop_neighbors(
        self,
        node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        *,
        neighbor_ty: Optional[NeighborType] = None,
    ):
        """
        get historical neighbors of nodes in node_ids at the first hop with max_num_neighbors as the maximal number of neighbors (make the computation feasible)
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :return:
        """
        # three lists to store the first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        (
            nodes_neighbor_ids_list,
            nodes_edge_ids_list,
            nodes_neighbor_times_list,
            nodes_neighbor_sign_list,
        ) = ([], [], [], [])
        # get the temporal neighbors at the first hop
        for idx, (node_id, node_interact_time) in enumerate(
            zip(node_ids, node_interact_times)
        ):
            # find neighbors that interacted with node_id before time node_interact_time
            (
                node_neighbor_ids,
                node_edge_ids,
                node_neighbor_times,
                node_neighbor_sign,
                _,
            ) = self.find_neighbors_before(
                node_id=node_id,
                interact_time=node_interact_time,
                return_sampled_probabilities=False,
                neighbor_ty=neighbor_ty,
            )
            nodes_neighbor_ids_list.append(node_neighbor_ids)
            nodes_edge_ids_list.append(node_edge_ids)
            nodes_neighbor_times_list.append(node_neighbor_times)
            nodes_neighbor_sign_list.append(node_neighbor_sign)

        return (
            nodes_neighbor_ids_list,
            nodes_edge_ids_list,
            nodes_neighbor_times_list,
            nodes_neighbor_sign_list,
        )

    def get_common_neighbors(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        *,
        neighbor_ty: Optional[NeighborType] = None,
    ):
        """
        提取节点对的历史共同邻居
        不但提取历史共同邻居，还要往前看
        """
        (
            src_nodes_neighbor_ids_list,
            src_nodes_edge_ids_list,
            src_nodes_neighbor_times_list,
            src_nodes_neighbor_sign_list,
        ) = ([], [], [], [])

        (
            dst_nodes_neighbor_ids_list,
            dst_nodes_edge_ids_list,
            dst_nodes_neighbor_times_list,
            dst_nodes_neighbor_sign_list,
        ) = ([], [], [], [])

        for idx, (src_node_id, dst_node_id, interact_time) in enumerate(
            zip(src_node_ids, dst_node_ids, node_interact_times)
        ):
            # find neighbors that interacted with node_id before time node_interact_time
            (
                src_node_neighbor_ids,
                src_node_edge_ids,
                src_node_neighbor_times,
                src_node_neighbor_sign,
                _,
            ) = self.find_neighbors_before(
                node_id=src_node_id,
                interact_time=interact_time,
                return_sampled_probabilities=False,
                neighbor_ty=neighbor_ty,
            )
            (
                dst_node_neighbor_ids,
                dst_node_edge_ids,
                dst_node_neighbor_times,
                dst_node_neighbor_sign,
                _,
            ) = self.find_neighbors_before(
                node_id=dst_node_id,
                interact_time=interact_time,
                return_sampled_probabilities=False,
                neighbor_ty=neighbor_ty,
            )

            common_neighbors = common_positions_np(
                src_node_neighbor_ids, dst_node_neighbor_ids
            )
            if len(common_neighbors) == 0:
                # 退回普通采样
                src_nodes_neighbor_ids_list.append(src_node_neighbor_ids)
                src_nodes_edge_ids_list.append(src_node_edge_ids)
                src_nodes_neighbor_times_list.append(src_node_neighbor_times)
                src_nodes_neighbor_sign_list.append(src_node_neighbor_sign)

                dst_nodes_neighbor_ids_list.append(dst_node_neighbor_ids)
                dst_nodes_edge_ids_list.append(dst_node_edge_ids)
                dst_nodes_neighbor_times_list.append(dst_node_neighbor_times)
                dst_nodes_neighbor_sign_list.append(dst_node_neighbor_sign)

            else:
                common_set = set(common_neighbors.keys())

                src_idxs = []
                dst_idxs = []
                for v, (src_pos, dst_pos) in common_neighbors.items():
                    for idx in src_pos:
                        start = max(0, idx - self.common_neighbors_look_forward)
                        # 往前找第一个公共节点或边界
                        for left in range(idx - 1, start - 1, -1):
                            if left < 0 or src_node_neighbor_ids[left] in common_set:
                                start = left + 1
                                break
                        src_idxs.append(np.arange(start, idx + 1, dtype=np.int32))

                    for idx in dst_pos:
                        start = max(0, idx - self.common_neighbors_look_forward)

                        for left in range(idx - 1, start - 1, -1):
                            if left < 0 or dst_node_neighbor_ids[left] in common_set:
                                start = left + 1
                                break
                        dst_idxs.append(np.arange(start, idx + 1, dtype=np.int32))

                src_idxs = (
                    np.concatenate(src_idxs)
                    if src_idxs
                    else np.array([], dtype=np.int64)
                )
                dst_idxs = (
                    np.concatenate(dst_idxs)
                    if dst_idxs
                    else np.array([], dtype=np.int64)
                )

                src_nodes_neighbor_ids_list.append(src_node_neighbor_ids[src_idxs])
                src_nodes_edge_ids_list.append(src_node_edge_ids[src_idxs])
                src_nodes_neighbor_times_list.append(src_node_neighbor_times[src_idxs])
                src_nodes_neighbor_sign_list.append(src_node_neighbor_sign[src_idxs])

                dst_nodes_neighbor_ids_list.append(dst_node_neighbor_ids[dst_idxs])
                dst_nodes_edge_ids_list.append(dst_node_edge_ids[dst_idxs])
                dst_nodes_neighbor_times_list.append(dst_node_neighbor_times[dst_idxs])
                dst_nodes_neighbor_sign_list.append(dst_node_neighbor_sign[dst_idxs])

        return (
            src_nodes_neighbor_ids_list,
            src_nodes_edge_ids_list,
            src_nodes_neighbor_times_list,
            src_nodes_neighbor_sign_list,
            dst_nodes_neighbor_ids_list,
            dst_nodes_edge_ids_list,
            dst_nodes_neighbor_times_list,
            dst_nodes_neighbor_sign_list,
        )

    def reset_random_state(self):
        """
        reset the random state by self.seed
        :return:
        """
        self.random_state = np.random.RandomState(self.seed)


def get_neighbor_sampler(
    data: Data,
    sample_neighbor_strategy: str = "uniform",
    time_scaling_factor: float = 0.0,
    seed: int = None,
    common_neighbor_look_forward: int = 2,
):
    """
    get neighbor sampler
    :param data: Data
    :param sample_neighbor_strategy: str, how to sample historical neighbors, 'uniform', 'recent', or 'time_interval_aware''
    :param time_scaling_factor: float, a hyper-parameter that controls the sampling preference with time interval,
    a large time_scaling_factor tends to sample more on recent links, this parameter works when sample_neighbor_strategy == 'time_interval_aware'
    :param seed: int, random seed
    :return:
    """
    max_node_id = max(data.src_node_ids.max(), data.dst_node_ids.max())
    # the adjacency vector stores edges for each node (source or destination), undirected
    # adj_list, list of list, where each element is a list of triple tuple (node_id, edge_id, timestamp)
    # the list at the first position in adj_list is empty
    adj_list: Dict[int, List[Neighbor]] = {idx: [] for idx in range(max_node_id + 1)}
    for (
        src_node_id,
        dst_node_id,
        edge_id,
        node_interact_time,
        node_interact_sign,
    ) in zip(
        data.src_node_ids,
        data.dst_node_ids,
        data.edge_ids,
        data.node_interact_times,
        data.node_interact_sign,
    ):
        adj_list[src_node_id].append(
            Neighbor(
                dst_node_id,
                edge_id,
                node_interact_time,
                node_interact_sign,
                NeighborType.OutcomeNeighbor,
            )
        )
        adj_list[dst_node_id].append(
            Neighbor(
                src_node_id,
                edge_id,
                node_interact_time,
                node_interact_sign,
                NeighborType.IncomeNeighbor,
            )
        )

    return DirectedNeighborSampler(
        adj_list=adj_list,
        sample_neighbor_strategy=sample_neighbor_strategy,
        time_scaling_factor=time_scaling_factor,
        seed=seed,
        common_neighbor_look_forward=common_neighbor_look_forward,
    )
