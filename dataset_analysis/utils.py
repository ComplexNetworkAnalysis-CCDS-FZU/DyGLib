from enum import Enum
import os


class Event:
    def __init__(self, src: str, dst: str, sign: int, timestamp: float):
        self.src = src
        self.dst = dst
        self.sign = sign
        self.timestamp = timestamp


class OrderMode(Enum):
    TimestampFirst = 0
    SignFirst = 1


class EventStream:
    def __init__(
        self,
        dataset: str,
        ext: str = "csv",
        sep: str = ",",
        order_mode: OrderMode = OrderMode.TimestampFirst,
        skip_head: bool = True,
    ):
        self.sep = sep
        self.order_mode = order_mode

        dataset_local = f"./DG_data/{dataset}/{dataset}.{ext}"
        assert os.path.exists(dataset_local), f"File Not Exist {dataset_local}"

        with open(dataset_local, "r", encoding="utf-8") as file:
            if skip_head:
                file.readline()
            self.lines = file.readlines()
        # 过滤0标签
        self.lines = filter(
            lambda x: int(
                x.split(self.sep)[2 if self.order_mode == OrderMode.SignFirst else 3]
            )
            != 0,
            self.lines,
        )
        # 排序
        self.lines = sorted(
            self.lines,
            key=lambda v: float(
                v.split(self.sep)[3 if self.order_mode == OrderMode.SignFirst else 2]
            ),
        )
        self.iter = iter(self.lines)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            line = next(self.iter)
        except StopIteration:
            raise
        else:
            split_items = [item.strip() for item in line.strip().split(sep=self.sep)]

            src = split_items[0]
            dst = split_items[1]

            sign = int(
                split_items[2]
                if self.order_mode == OrderMode.SignFirst
                else split_items[3]
            )
            sign = sign // abs(sign)
            timestamp = float(
                split_items[3]
                if self.order_mode == OrderMode.SignFirst
                else split_items[2]
            )

            return Event(src, dst, sign, timestamp)

    def __len__(self):
        return self.lines.__len__()
