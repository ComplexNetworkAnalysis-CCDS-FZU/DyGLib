"""test_cne_off.py — CNE-off（.CNE-D）探针接口自检（2026-09-17；纯 CPU，本机可跑）。

背景：Paper 6c7a 四 / 用户 2026-09-17 批准 —— 实现选项 (a)：CNE 通道特征置零
（张量维度 / 模型参数结构不变，切"信息"不切"结构"；默认开启 ⇒ 零行为变更）。

自检项：
① 语义：CNE 关闭 ⇒ 共现特征【严格全零】；形状 (B,L,feat_dim)、dtype float32 与开启一致；
② 兼容：CNE 默认开启 ⇒ 输出非零（原计算路径未动）；
③ 接线：SignDyGFormer(module_common_neighbor_encoder=False) 能到达编码器；
④ 配置层：result_save_name —— 默认无 .CNE-D；关闭时以 .CNE-D 结尾；
   --ablation 的 .NN-Best.LF-Best 规则不受影响；CLI 开关 --no-module-common-neighbor-encoder 可解析。

用法（仓库根目录）：python tools/verify/test_cne_off.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from models.NeighborInteractEncoder import (  # noqa: E402
    EncodeType,
    NeighborCooccurrenceEncoder,
)
from models.SignDyGFormer import SignDyGFormer  # noqa: E402
from utils.load_configs import SignPredictArgs  # noqa: E402

FAILS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = "[OK]" if cond else "[FAIL]"
    print(f"{tag} {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def make_inputs(seed: int = 0):
    rng = np.random.RandomState(seed)
    b, ls, ld = 3, 5, 4
    return dict(
        src_ids=rng.randint(1, 20, size=b),
        dst_ids=rng.randint(1, 20, size=b),
        src_padded_nodes_neighbor_ids=rng.randint(0, 12, size=(b, ls)),
        dst_padded_nodes_neighbor_ids=rng.randint(0, 12, size=(b, ld)),
        src_padded_nodes_neighbor_sign=rng.choice([-1, 1], size=(b, ls)).astype(np.int64),
        dst_padded_nodes_neighbor_sign=rng.choice([-1, 1], size=(b, ld)).astype(np.int64),
    )


def run_encoder(cne_on: bool):
    enc = NeighborCooccurrenceEncoder(
        neighbor_co_occurrence_feat_dim=8,
        device="cpu",
        module_common_neighbor_encoder=cne_on,
    )
    return enc.forward(**make_inputs(), sample_type=EncodeType.CoOccurredNeighbor)


def main() -> int:
    inputs = make_inputs()

    # ① / ② encoder 语义与兼容
    src_on, dst_on = run_encoder(cne_on=True)
    src_off, dst_off = run_encoder(cne_on=False)

    check(
        "CNE 开启：输出非零（原路径）",
        bool(torch.count_nonzero(src_on) > 0 and torch.count_nonzero(dst_on) > 0),
        f"src nnz={int(torch.count_nonzero(src_on))}",
    )
    check(
        "CNE 关闭：src/dst 特征严格全零",
        int(torch.count_nonzero(src_off)) == 0 and int(torch.count_nonzero(dst_off)) == 0,
    )
    check(
        "形状一致 (B,L,feat_dim)",
        tuple(src_on.shape) == tuple(src_off.shape) == (3, 5, 8)
        and tuple(dst_on.shape) == tuple(dst_off.shape) == (3, 4, 8),
        f"src {tuple(src_off.shape)}, dst {tuple(dst_off.shape)}",
    )
    check(
        "dtype 一致 (float32)",
        src_off.dtype == torch.float32 and dst_off.dtype == torch.float32,
        f"{src_off.dtype}",
    )

    # ③ SignDyGFormer 接线
    try:
        model = SignDyGFormer(
            node_raw_features=np.random.RandomState(0).rand(4, 6).astype(np.float32),
            edge_raw_features=np.random.RandomState(1).rand(8, 3).astype(np.float32),
            neighbor_sampler=None,
            time_feat_dim=8,
            channel_embedding_dim=8,
            patch_size=1,
            num_layers=1,
            num_heads=1,
            dropout=0.0,
            max_input_sequence_length=4,
            device="cpu",
            module_common_neighbor_encoder=False,
        )
        enc = model.neighbor_co_occurrence_encoder
        ok = enc.module_common_neighbor_encoder is False
        detail = f"encoder.module_common_neighbor_encoder={enc.module_common_neighbor_encoder}"
    except Exception as exc:  # pragma: no cover
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    check("SignDyGFormer 构造参数接线", ok, detail)

    # ④ 配置层：命名
    name_default = SignPredictArgs().result_save_name
    name_off = SignPredictArgs(module_common_neighbor_encoder=False).result_save_name
    name_abl_off = SignPredictArgs(
        ablation=True, module_common_neighbor_encoder=False
    ).result_save_name
    name_abl_on = SignPredictArgs(ablation=True).result_save_name

    check("默认命名不含 .CNE-D（零行为变更）", ".CNE-D" not in name_default, name_default)
    check("关闭命名以 .CNE-D 结尾", name_off.endswith(".CNE-D"), name_off)
    check(
        "ablation 规则不受影响且可叠加",
        ".NN-Best.LF-Best" in name_abl_off
        and name_abl_off.endswith(".CNE-D")
        and ".NN-Best.LF-Best" in name_abl_on
        and ".CNE-D" not in name_abl_on,
        name_abl_off,
    )

    # ④ CLI 开关（pydantic_argparse 生成 --no-module-common-neighbor-encoder）
    try:
        import pydantic_argparse

        parser = pydantic_argparse.ArgumentParser(SignPredictArgs)
        saved = sys.argv
        try:
            sys.argv = ["test_cne_off", "--no-module-common-neighbor-encoder"]
            parsed = parser.parse_typed_args()
        finally:
            sys.argv = saved
        ok = parsed.module_common_neighbor_encoder is False
        detail = f"parsed={parsed.module_common_neighbor_encoder}"
    except Exception as exc:  # pragma: no cover
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    check("CLI 开关 --no-module-common-neighbor-encoder", ok, detail)

    if FAILS:
        print(f"\n[FAILED] {len(FAILS)} 项未通过：{FAILS}")
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
