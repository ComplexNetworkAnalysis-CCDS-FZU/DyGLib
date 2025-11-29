from enum import Enum
from typing import Counter, Dict, List, Tuple
import networkx as nx
from tqdm import tqdm

from utils import OrderMode, EventStream, Event


class TriadStatus(Enum):
    Balance = 0
    UnBalance = 1

    @staticmethod
    def from_neg_edge(num: int):
        if num % 2 == 0:
            return TriadStatus.Balance
        else:
            return TriadStatus.UnBalance


def triad_detect(event_stream: EventStream):
    G = nx.MultiGraph()
    triSign: Dict[frozenset, List[Tuple[int, float, TriadStatus]]] = {}
    for event in tqdm(event_stream):

        # 插入
        G.add_edge(event.src, event.dst, sign=event.sign, timestamp=event.timestamp)

        # 遍历历史共邻居
        common = nx.common_neighbors(G, event.src, event.dst)
        u, v, sign, time = event.src, event.dst, event.sign, event.timestamp

        for w in common:
            tri = frozenset({u, v, w})
            e1, e2 = (
                G[u][w][len(G[u][w]) - 1]["sign"],
                G[v][w][len(G[v][w]) - 1]["sign"],
            )
            new_neg = (
                (1 if sign == -1 else 0)
                + (1 if e1 == -1 else 0)
                + (1 if e2 == -1 else 0)
            )

            if tri not in triSign.keys():
                triSign[tri] = []

            triSign[tri].append((new_neg, time, TriadStatus.from_neg_edge(new_neg)))

    return triSign


def get_tri_balance_unbalance_time(
    tri: frozenset, seq: List[Tuple[int, float, TriadStatus]]
):

    if not seq:
        return [], []

    seq_sorted = sorted(seq, key=lambda x: x[1])

    stable_duration: List[float] = []
    us_interval: List[float] = []

    prev_t, prev_stat = seq_sorted[0][1], seq_sorted[0][2]

    stable_start: float | None = None if prev_stat != TriadStatus.Balance else prev_t
    start_us: float | None = None if prev_stat != TriadStatus.UnBalance else prev_t

    for _, t, stat in seq_sorted[1:]:

        # 反转
        if prev_stat == TriadStatus.Balance and stat == TriadStatus.UnBalance:
            # 翻转为非稳定
            # 如果有先前的稳定开始时间，那开始
            if stable_start is not None:
                interval = t - stable_start
                stable_duration.append(interval)
            stable_start = None
            start_us = t
        elif prev_stat == TriadStatus.UnBalance and stat == TriadStatus.Balance:
            # 反转为稳定
            # 如果有先前不稳定开始时间，计算
            if start_us is not None:
                interval = t - start_us
                us_interval.append(interval)

            start_us = None
            stable_start = t

    return stable_duration, us_interval

def balance_flip_stats(data: Dict[frozenset, List[Tuple[int, float, TriadStatus]]]):
    cnt = {
        'flip_SU':0,
        'flip_US':0,
        "any_flip":0,
        "always_S":0,
        "always_U":0


    }         # 各类原始计数
    for seq in data.values():
        if not seq:
            continue
        first = seq[0][2]
        last  = seq[-1][2]
        if first != last:               # 曾经翻转
            cnt['flip_SU'] += int(first==TriadStatus.Balance and last==TriadStatus.UnBalance)
            cnt['flip_US'] += int(first==TriadStatus.UnBalance and last==TriadStatus.Balance)
            cnt['any_flip'] += 1
        else:                           # 始终不变
            cnt['always_S'] += int(first==TriadStatus.Balance)
            cnt['always_U'] += int(first==TriadStatus.UnBalance)

    total = len(data)
    return {
        'flip_SU_ratio': cnt['flip_SU'] / total,
        'flip_US_ratio': cnt['flip_US'] / total,
        'any_flip_ratio': cnt['any_flip'] / total,
        'always_S_ratio': cnt['always_S'] / total,
        'always_U_ratio': cnt['always_U'] / total,
        'never_change_ratio': (cnt['always_S'] + cnt['always_U']) / total
    }

def flip_direction_counts(data: Dict[frozenset, List[Tuple[int, float, TriadStatus]]]):
    """返回 {'SU': 次数, 'US': 次数}  仅统计实际发生的翻转"""
    directional = Counter({'SU': 0, 'US': 0})
    for seq in data.values():
        if len(seq) < 2:          # 无翻转
            continue
        prev_st = seq[0][2]
        for _, _, st in seq[1:]:
            # 发生反转
            if st != prev_st:              
                if st == TriadStatus.Balance:
                    directional["US"] += 1
                if st == TriadStatus.UnBalance:
                    directional["SU"] +=1
            prev_st = st
    return directional


if __name__ == "__main__":
    dataset = "BitcoinOTC"

    event_steam = EventStream(
        dataset, order_mode=OrderMode.SignFirst, skip_head=False
    )
    triad = triad_detect(event_steam)

    statical = balance_flip_stats(triad)

    counts = flip_direction_counts(triad)
    total_flips = counts['SU'] + counts['US']
    su_ratio = counts['SU'] / total_flips
    us_ratio = counts['US'] / total_flips

    statical["balance_flip_rate"] = su_ratio
    statical["unbalance_flip_rate"] = us_ratio

    print(statical)

    for_save = {
        f"{k}"[10:-1]: [{"time": t, "state": state.name} for _, t, state in v]
        for k, v in triad.items()
    }

    triad_rev = {
        f"{k}"[10:-1]: {"balance_duration": v[0], "unbalance_duration": v[1]}
        for k, v in [
            (k1, get_tri_balance_unbalance_time(k1, v1)) for k1, v1 in triad.items()
        ]
    }

    with open(f"{dataset}-triad.json", "w", encoding="utf-8") as f:
        import json
        try:
            json.dump({"statical":statical,"triad_record": for_save, "triad_duration": triad_rev}, f, indent=4)
        except e:
            print(e)