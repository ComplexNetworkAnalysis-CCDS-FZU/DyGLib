"""
验证批次内反向边假设:
  "一个批次内不会出现重复的反向负样本"
即对任意 batch 中的边 u→v，其反向 v→u 在 batch 内至多出现一次。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import argparse
from utils.DataLoader import get_link_prediction_data


def check_batch_reverse(
    dataset_name: str,
    batch_size: int = 200,
    seed: int = 2026,
):
    (_, _, _, train_data, val_data, test_data, _, _) = get_link_prediction_data(
        dataset_name=dataset_name, val_ratio=0.15, test_ratio=0.15
    )

    rng = np.random.RandomState(seed)

    for split_name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        N = len(data.src_node_ids)
        indices = np.arange(N)
        rng.shuffle(indices)

        total_pairs = 0
        rev_pairs = 0
        rev_multi = 0  # 同一 batch 内反向出现 >1 次

        for start in range(0, N, batch_size):
            batch_idx = indices[start : start + batch_size]
            src = data.src_node_ids[batch_idx]
            dst = data.dst_node_ids[batch_idx]

            # 构建 (src,dst) → index 映射
            edge_set = {}
            for i in range(len(src)):
                key = (src[i], dst[i])
                edge_set[key] = edge_set.get(key, 0) + 1

            for (s, d), cnt in edge_set.items():
                total_pairs += 1
                rev_key = (d, s)
                if rev_key in edge_set:
                    rev_pairs += 1
                    rev_multi += max(0, edge_set[rev_key] - 1)

        rev_ratio = rev_pairs / total_pairs if total_pairs > 0 else 0
        print(
            f"[{split_name:5s}] batch_size={batch_size:4d} | "
            f"total_pairs={total_pairs:6d} | rev_pairs={rev_pairs:5d} "
            f"({rev_ratio:.4f}) | rev_multi={rev_multi}"
        )
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="BitcoinAlpha")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 128, 200, 400, 800])
    args = parser.parse_args()

    for ds in [args.dataset]:
        print(f"\n=== {ds} ===")
        for bs in args.batch_sizes:
            check_batch_reverse(ds, bs)
