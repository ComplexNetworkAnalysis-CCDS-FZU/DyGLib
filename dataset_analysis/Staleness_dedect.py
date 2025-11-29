# 检测数据集是否有僵化情况

import os
from enum import Enum
from typing import Dict ,List
import numpy as np
from tqdm import tqdm

from utils import EventStream, OrderMode


class InteractRecord(object):
    def __init__(self):
        self.interact_to_time:List[float] = []
        self. interact_from_time:List[float] = []

    def get_interval(self):
        return list(np.diff(self.interact_to_time)),list(np.diff(self.interact_from_time))
    
    def add_to(self,time:float):
        self.interact_to_time.append(time)

    def add_from(self,time:float):
        self.interact_from_time.append(time)
    


def staleness_detect(event_stream:EventStream):

    node_activate_map :Dict[str,InteractRecord] = dict()


    for event in tqdm(event_stream):

        # print(split_items)

        src = event.src
        dst = event.dst
        time = event.timestamp

        # 更新交互双方节点的更新序列

        if src not in node_activate_map.keys():
            node_activate_map[src] = InteractRecord()
        if dst not in node_activate_map.keys():
            node_activate_map[dst] = InteractRecord()

        node_activate_map[src].add_to(time)
        node_activate_map[dst].add_from(time)

    node_active_interval = {}
    # 计算各个节点的时间间隔
    for node, active_time in node_activate_map.items():
        to_interval,from_interval = active_time.get_interval()
        node_active_interval[node] = {
            "to_interval":to_interval,
            "from_interval":from_interval
        }

    return node_active_interval

if __name__ == "__main__":
    dataset="BitcoinAlpha"

    event_steam = EventStream(dataset,order_mode=OrderMode.SignFirst,skip_head=False)
    interval = staleness_detect(event_steam)

    with open(f"{dataset}.json","w",encoding="utf-8") as f:
        import json
        json.dump(interval,f,indent=4)