"""带方向的历史交互邻居采样器"""

from enum import Enum
from typing import Dict, List, Optional, OrderedDict, Tuple
import numpy as np
import torch
from tqdm import tqdm

from utils.DataLoader import Data
from utils.profiler import Profiler
from utils import accel as _accel


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


def common_neighbor_location(
    src_neighbor: np.ndarray,
    dst_neighbor: np.ndarray,
    *,
    repeat_aware: bool = False,
    src: Optional[int] = None,
    dst: Optional[int] = None,
):
    """
    a, b : 1-D numpy array
    return : dict{value: (idx_a, idx_b)}
    """

    # 交集 & 组装
    # 修复（2026-09-11 用户决策）：真集合交集——与论文 §3.2  C=N_u∩N_v  语义对齐。
    # 原 assume_unique=True 在历史含重复 id（重复边）时按“排序后相邻相等”计数，会把
    # “单侧重复≥2、另一侧 0 次”的邻居误判为共同邻居（伪 CN）→ 采样锚点/窗口偏差。
    # 量化与决策：docs/ANALYSIS_CN_PSEUDO_INTERSECT.md
    common_vals = np.intersect1d(src_neighbor, dst_neighbor)
    # 两个分别表示在src的位置和在dst的位置
    aware_nodes = {}
    for v in common_vals:
        # np.where 仍是瓶颈，但可用 np.argwhere 或 numba 加速
        src_pos = np.where(src_neighbor == v)[0]
        dst_pos = np.where(dst_neighbor == v)[0]
        aware_nodes[int(v)] = (src_pos, dst_pos)
    if repeat_aware:

        assert (
            src is not None and dst is not None
        ), "重复交互感知采样需要提供对向节点信息"
        dst_in_src = aware_nodes.get(int(dst), (np.array([], dtype=int), None))[0]
        src_in_dst = aware_nodes.get(int(src), (None, np.array([], dtype=int)))[1]

        if len(dst_in_src) != 0:
            aware_nodes[int(dst)] = (dst_in_src, np.array([], dtype=int))

        if len(src_in_dst) != 0:
            aware_nodes[int(src)] = (np.array([], dtype=int), src_in_dst)

    return aware_nodes


class DirectedNeighborSampler:

    def __init__(
        self,
        adj_list: Dict[int, List[Neighbor]],
        sample_neighbor_strategy: str = "uniform",
        time_scaling_factor: float = 0.0,
        seed: int = None,
        common_neighbor_look_forward: int = 5,
        module_repeat_aware_sampler: bool = False,
        module_common_neighbor_sampler: bool = True,
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
        self.module_repeat_aware_sampler = module_repeat_aware_sampler
        self.module_common_neighbor_sampler = module_common_neighbor_sampler

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
            prof = Profiler()
            # ---- M4 加速接缝（默认启用；--no-accel / SIGNDYG_ACCEL=0 关闭）----
            # 与下方原路径逐位一致（K1 bit-exact 门禁验证）；启用时内核内部完成
            # searchsorted 截断 + CN + RAS + look-forward + concat/sort，返回升序下标。
            if _accel.on:
                _nb = self.nodes_neighbors_selector(neighbor_ty)
                with prof.timer("Sampling: Accel K1"):
                    src_sel, dst_sel = _accel.kernel.core_sample(
                        np.ascontiguousarray(_nb.ids[src_node_id], dtype=np.int64),
                        np.ascontiguousarray(_nb.times[src_node_id], dtype=np.float64),
                        np.ascontiguousarray(_nb.ids[dst_node_id], dtype=np.int64),
                        np.ascontiguousarray(_nb.times[dst_node_id], dtype=np.float64),
                        float(interact_time),
                        k=self.common_neighbors_look_forward,
                        repeat_aware=self.module_repeat_aware_sampler,
                        src_id=int(src_node_id),
                        dst_id=int(dst_node_id),
                    )
                src_nodes_neighbor_ids_list.append(_nb.ids[src_node_id][src_sel])
                src_nodes_edge_ids_list.append(_nb.edges_ids[src_node_id][src_sel])
                src_nodes_neighbor_times_list.append(_nb.times[src_node_id][src_sel])
                src_nodes_neighbor_sign_list.append(_nb.signs[src_node_id][src_sel])
                dst_nodes_neighbor_ids_list.append(_nb.ids[dst_node_id][dst_sel])
                dst_nodes_edge_ids_list.append(_nb.edges_ids[dst_node_id][dst_sel])
                dst_nodes_neighbor_times_list.append(_nb.times[dst_node_id][dst_sel])
                dst_nodes_neighbor_sign_list.append(_nb.signs[dst_node_id][dst_sel])
                continue
            # find neighbors that interacted with node_id before time node_interact_time

            with prof.timer("Sampling: History Neighbor"):
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

            with prof.timer("Sampling: Common Neighbor"):
                common_neighbors = common_neighbor_location(
                    src_node_neighbor_ids,
                    dst_node_neighbor_ids,
                    repeat_aware=self.module_repeat_aware_sampler,
                    src=src_node_id,
                    dst=dst_node_id,
                )
                # ---- RAS（论文 §3.2，修复 2026-09-09）：R 锚点扩充 ----
                # 预测 (u,v) 时，u 与 v 的【直接历史】（v∈N(u) 或 u∈N(v)）应作为额外采样锚点，
                # 使直接重复交互进入 CNAS 窗口。旧实现只在“自环”时改写已有键（几乎永不触发）；
                # 此处改为：有直接历史即【新增】锚点（普通重复对即可触发）。
                # 说明：look_forward_sampling 支持单侧锚点（src_pos 只生成 src 窗、dst_pos 只生成 dst 窗）。
                if self.module_repeat_aware_sampler:
                    r_src_pos = np.where(src_node_neighbor_ids == dst_node_id)[0]
                    if len(r_src_pos) > 0:
                        prev = common_neighbors.get(int(dst_node_id))
                        prev_src = prev[0] if prev is not None else np.array([], dtype=int)
                        prev_dst = prev[1] if prev is not None else np.array([], dtype=int)
                        common_neighbors[int(dst_node_id)] = (
                            np.union1d(prev_src, r_src_pos),
                            prev_dst,
                        )
                    r_dst_pos = np.where(dst_node_neighbor_ids == src_node_id)[0]
                    if len(r_dst_pos) > 0:
                        prev = common_neighbors.get(int(src_node_id))
                        prev_src = prev[0] if prev is not None else np.array([], dtype=int)
                        prev_dst = prev[1] if prev is not None else np.array([], dtype=int)
                        common_neighbors[int(src_node_id)] = (
                            prev_src,
                            np.union1d(prev_dst, r_dst_pos),
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
                with prof.timer("Sampling: Looking Forward"):

                    src_idxs, dst_idxs = look_forward_sampling(
                        src_node_neighbor_ids,
                        dst_node_neighbor_ids,
                        common_neighbors,
                        self.common_neighbors_look_forward,
                    )

                with prof.timer("Sampling: Concat & Sort"):
                    src_idxs = (
                        np.concatenate(src_idxs)
                        if src_idxs
                        else np.array([], dtype=np.int64)
                    )
                    src_idxs.sort()
                    dst_idxs = (
                        np.concatenate(dst_idxs)
                        if dst_idxs
                        else np.array([], dtype=np.int64)
                    )
                    dst_idxs.sort()

                assert np.all(
                    np.diff(src_node_neighbor_times[src_idxs]) >= 0
                ), "src历史邻居采样序列不是升序的"
                assert np.all(
                    np.diff(dst_node_neighbor_times[dst_idxs]) >= 0
                ), "dst历史邻居采样序列不是升序的"

                src_nodes_neighbor_ids_list.append(src_node_neighbor_ids[src_idxs])
                src_nodes_edge_ids_list.append(src_node_edge_ids[src_idxs])
                src_nodes_neighbor_times_list.append(src_node_neighbor_times[src_idxs])
                src_nodes_neighbor_sign_list.append(src_node_neighbor_sign[src_idxs])

                dst_nodes_neighbor_ids_list.append(dst_node_neighbor_ids[dst_idxs])
                dst_nodes_edge_ids_list.append(dst_node_edge_ids[dst_idxs])
                dst_nodes_neighbor_times_list.append(dst_node_neighbor_times[dst_idxs])
                dst_nodes_neighbor_sign_list.append(dst_node_neighbor_sign[dst_idxs])
            # prof.report()
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

    def get_directed_common_neighbors(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
    ):
        """
        有向版本的共同邻居采样，分别处理出入邻居序列。
        对节点对 (u, v):
          - src_out ∩ dst_in → Type A (u→k→v)
          - src_in ∩ dst_out → Type B (v→k→u)
        每组独立进行 look_forward_sampling。
        :return: 8 个 list — (src_out_ids, src_out_edges, src_out_times, src_out_signs,
                                  src_in_ids,  src_in_edges,  src_in_times,  src_in_signs,
                                  dst_out_ids, dst_out_edges, dst_out_times, dst_out_signs,
                                  dst_in_ids,  dst_in_edges,  dst_in_times,  dst_in_signs)
        """
        src_out_ids_list, src_out_edges_list, src_out_times_list, src_out_signs_list = (
            [], [], [], []
        )
        src_in_ids_list, src_in_edges_list, src_in_times_list, src_in_signs_list = (
            [], [], [], []
        )
        dst_out_ids_list, dst_out_edges_list, dst_out_times_list, dst_out_signs_list = (
            [], [], [], []
        )
        dst_in_ids_list, dst_in_edges_list, dst_in_times_list, dst_in_signs_list = (
            [], [], [], []
        )

        for src_id, dst_id, interact_time in zip(
            src_node_ids, dst_node_ids, node_interact_times
        ):
            # ---- 采样四个方向的历史邻居 ----
            (src_out_ids, src_out_edges, src_out_times, src_out_signs, _) = (
                self.find_neighbors_before(
                    node_id=src_id,
                    interact_time=interact_time,
                    neighbor_ty=NeighborType.OutcomeNeighbor,
                )
            )
            (src_in_ids, src_in_edges, src_in_times, src_in_signs, _) = (
                self.find_neighbors_before(
                    node_id=src_id,
                    interact_time=interact_time,
                    neighbor_ty=NeighborType.IncomeNeighbor,
                )
            )
            (dst_out_ids, dst_out_edges, dst_out_times, dst_out_signs, _) = (
                self.find_neighbors_before(
                    node_id=dst_id,
                    interact_time=interact_time,
                    neighbor_ty=NeighborType.OutcomeNeighbor,
                )
            )
            (dst_in_ids, dst_in_edges, dst_in_times, dst_in_signs, _) = (
                self.find_neighbors_before(
                    node_id=dst_id,
                    interact_time=interact_time,
                    neighbor_ty=NeighborType.IncomeNeighbor,
                )
            )

            # ---- Type A: src_out ∩ dst_in (u→k→v) ----
            type_a_common = common_neighbor_location(
                src_out_ids,
                dst_in_ids,
                repeat_aware=self.module_repeat_aware_sampler,
                src=src_id,
                dst=dst_id,
            )
            if len(type_a_common) > 0:
                src_out_idxs_a, dst_in_idxs_a = look_forward_sampling(
                    src_out_ids,
                    dst_in_ids,
                    type_a_common,
                    self.common_neighbors_look_forward,
                )
                src_out_idxs_a = (
                    np.concatenate(src_out_idxs_a)
                    if src_out_idxs_a
                    else np.array([], dtype=np.int64)
                )
                src_out_idxs_a.sort()
                dst_in_idxs_a = (
                    np.concatenate(dst_in_idxs_a)
                    if dst_in_idxs_a
                    else np.array([], dtype=np.int64)
                )
                dst_in_idxs_a.sort()
            else:
                src_out_idxs_a = np.arange(len(src_out_ids), dtype=np.int64)
                dst_in_idxs_a = np.arange(len(dst_in_ids), dtype=np.int64)

            # ---- Type B: src_in ∩ dst_out (v→k→u) ----
            type_b_common = common_neighbor_location(
                src_in_ids,
                dst_out_ids,
                repeat_aware=self.module_repeat_aware_sampler,
                src=src_id,
                dst=dst_id,
            )
            if len(type_b_common) > 0:
                src_in_idxs_b, dst_out_idxs_b = look_forward_sampling(
                    src_in_ids,
                    dst_out_ids,
                    type_b_common,
                    self.common_neighbors_look_forward,
                )
                src_in_idxs_b = (
                    np.concatenate(src_in_idxs_b)
                    if src_in_idxs_b
                    else np.array([], dtype=np.int64)
                )
                src_in_idxs_b.sort()
                dst_out_idxs_b = (
                    np.concatenate(dst_out_idxs_b)
                    if dst_out_idxs_b
                    else np.array([], dtype=np.int64)
                )
                dst_out_idxs_b.sort()
            else:
                src_in_idxs_b = np.arange(len(src_in_ids), dtype=np.int64)
                dst_out_idxs_b = np.arange(len(dst_out_ids), dtype=np.int64)

            # ---- 写入结果 ----
            src_out_ids_list.append(src_out_ids[src_out_idxs_a])
            src_out_edges_list.append(src_out_edges[src_out_idxs_a])
            src_out_times_list.append(src_out_times[src_out_idxs_a])
            src_out_signs_list.append(src_out_signs[src_out_idxs_a])

            src_in_ids_list.append(src_in_ids[src_in_idxs_b])
            src_in_edges_list.append(src_in_edges[src_in_idxs_b])
            src_in_times_list.append(src_in_times[src_in_idxs_b])
            src_in_signs_list.append(src_in_signs[src_in_idxs_b])

            dst_out_ids_list.append(dst_out_ids[dst_out_idxs_b])
            dst_out_edges_list.append(dst_out_edges[dst_out_idxs_b])
            dst_out_times_list.append(dst_out_times[dst_out_idxs_b])
            dst_out_signs_list.append(dst_out_signs[dst_out_idxs_b])

            dst_in_ids_list.append(dst_in_ids[dst_in_idxs_a])
            dst_in_edges_list.append(dst_in_edges[dst_in_idxs_a])
            dst_in_times_list.append(dst_in_times[dst_in_idxs_a])
            dst_in_signs_list.append(dst_in_signs[dst_in_idxs_a])

        return (
            src_out_ids_list, src_out_edges_list, src_out_times_list, src_out_signs_list,
            src_in_ids_list,  src_in_edges_list,  src_in_times_list,  src_in_signs_list,
            dst_out_ids_list, dst_out_edges_list, dst_out_times_list, dst_out_signs_list,
            dst_in_ids_list,  dst_in_edges_list,  dst_in_times_list,  dst_in_signs_list,
        )

    def history_neighbors_sampling(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        *,
        neighbor_ty: Optional[NeighborType] = None,
    ):
        if self.module_common_neighbor_sampler:
            return self.get_common_neighbors(
                src_node_ids=src_node_ids,
                dst_node_ids=dst_node_ids,
                node_interact_times=node_interact_times,
                neighbor_ty=neighbor_ty,
            )
        else:
            src_neighbor_ids, src_edge_ids, src_interact_time, src_interact_sign = (
                self.get_all_first_hop_neighbors(
                    node_ids=src_node_ids,
                    node_interact_times=node_interact_times,
                    neighbor_ty=neighbor_ty,
                )
            )
            dst_neighbor_ids, dst_edge_ids, dst_interact_time, dst_interact_sign = (
                self.get_all_first_hop_neighbors(
                    node_ids=dst_node_ids,
                    node_interact_times=node_interact_times,
                    neighbor_ty=neighbor_ty,
                )
            )
            return (
                src_neighbor_ids,
                src_edge_ids,
                src_interact_time,
                src_interact_sign,
                dst_neighbor_ids,
                dst_edge_ids,
                dst_interact_time,
                dst_interact_sign,
            )

    def history_neighbors_sampling_directed(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
    ):
        """
        有向符号图的历史邻居采样。
        为每个节点分别采样出邻居和入邻居，并通过有向共同邻居进行前瞻采样。
        :return: 8 个 list —
                 (src_out_ids, src_out_edges, src_out_times, src_out_signs,
                  src_in_ids,  src_in_edges,  src_in_times,  src_in_signs,
                  dst_out_ids, dst_out_edges, dst_out_times, dst_out_signs,
                  dst_in_ids,  dst_in_edges,  dst_in_times,  dst_in_signs)
        """
        if self.module_common_neighbor_sampler:
            return self.get_directed_common_neighbors(
                src_node_ids=src_node_ids,
                dst_node_ids=dst_node_ids,
                node_interact_times=node_interact_times,
            )
        else:
            # 关闭共邻居采样时，直接取全部出入邻居
            (
                src_out_ids, src_out_edges, src_out_times, src_out_signs,
            ) = self.get_all_first_hop_neighbors(
                node_ids=src_node_ids,
                node_interact_times=node_interact_times,
                neighbor_ty=NeighborType.OutcomeNeighbor,
            )
            (
                src_in_ids, src_in_edges, src_in_times, src_in_signs,
            ) = self.get_all_first_hop_neighbors(
                node_ids=src_node_ids,
                node_interact_times=node_interact_times,
                neighbor_ty=NeighborType.IncomeNeighbor,
            )
            (
                dst_out_ids, dst_out_edges, dst_out_times, dst_out_signs,
            ) = self.get_all_first_hop_neighbors(
                node_ids=dst_node_ids,
                node_interact_times=node_interact_times,
                neighbor_ty=NeighborType.OutcomeNeighbor,
            )
            (
                dst_in_ids, dst_in_edges, dst_in_times, dst_in_signs,
            ) = self.get_all_first_hop_neighbors(
                node_ids=dst_node_ids,
                node_interact_times=node_interact_times,
                neighbor_ty=NeighborType.IncomeNeighbor,
            )
            return (
                src_out_ids, src_out_edges, src_out_times, src_out_signs,
                src_in_ids,  src_in_edges,  src_in_times,  src_in_signs,
                dst_out_ids, dst_out_edges, dst_out_times, dst_out_signs,
                dst_in_ids,  dst_in_edges,  dst_in_times,  dst_in_signs,
            )

    def reset_random_state(self):
        """
        reset the random state by self.seed
        :return:
        """
        self.random_state = np.random.RandomState(self.seed)


def look_forward_sampling(
    src_neighbor_ids, dst_neighbor_ids, common_neighbors: dict, k
):
    pf = Profiler()

    with pf.timer("LF: All Common Neighbor "):
        src_all_common = np.sort(
            np.concatenate([v[0] for v in common_neighbors.values()])
        )
        dst_all_common = np.sort(
            np.concatenate([v[1] for v in common_neighbors.values()])
        )

    with pf.timer("LF-look forward"):
        src_idxs = []
        dst_idxs = []

        for v, (src_pos, dst_pos) in common_neighbors.items():
            for idx in src_pos:
                start = max(0, idx - k)
                left, right = 0, len(src_all_common)
                # 往前找第一个公共节点或边界
                while left < right:
                    mid = (left + right) // 2
                    if src_all_common[mid] < idx:
                        left = mid + 1
                    else:
                        right = mid
                if left > 0 and src_all_common[left - 1] >= start:
                    start = src_all_common[left - 1] + 1
                src_idxs.append(np.arange(start, idx + 1, dtype=np.int32))

            for idx in dst_pos:
                start = max(0, idx - k)

                left, right = 0, len(dst_all_common)
                # 往前找第一个公共节点或边界
                while left < right:
                    mid = (left + right) // 2
                    if dst_all_common[mid] < idx:
                        left = mid + 1
                    else:
                        right = mid
                if left > 0 and dst_all_common[left - 1] >= start:
                    start = dst_all_common[left - 1] + 1
                # 往前找第一个公共节点或边界
                dst_idxs.append(np.arange(start, idx + 1, dtype=np.int32))
    # pf.report()

    return src_idxs, dst_idxs


def find_boundary_numba(neighbor_ids, aware_flags, idx, k):
    start = max(0, idx - k)

    # 在 [start, idx) 范围内找最后一个感知节点
    for i in range(idx - 1, start - 1, -1):
        if aware_flags[i]:
            return i + 1  # 感知节点不纳入，从下一个开始

    return start


def get_neighbor_sampler(
    *,
    data: Data,
    sample_neighbor_strategy: str = "uniform",
    time_scaling_factor: float = 0.0,
    seed: int = None,
    common_neighbor_look_forward: int = 2,
    module_repeat_aware_sampler: bool = False,
    module_common_neighbor_sampler: bool = True,
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
        module_repeat_aware_sampler=module_repeat_aware_sampler,
        module_common_neighbor_sampler=module_common_neighbor_sampler,
    )
