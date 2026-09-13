"""
DirectSignDyGFormer 纯有向链路预测训练脚本。
任务: 给定 (src, dst, t)，预测有向边 u→v 是否存在。
样本: pos (真实边) / rev (批次内反向边) / neg (随机负采样)
"""
import logging
import json
import time
import os
import numpy as np
import warnings
import torch
import torch.nn as nn
from tqdm import tqdm

from models.DirectSignDyGFormer import DirectSignDyGFormer
from models.SignDyGFormer import SignDyGFormer
from models.DyGFormer import DyGFormer
from models.TGAT import TGAT
from models.GraphMixer import GraphMixer
from models.modules import MergeLayer
from losses import LinkLoss, LinkLossType, EdgeDirectionMode
from utils.metrics.linkPredict import get_link_prediction_metrics
from evaluate_models_utils import evaluate_directed_link_prediction
from utils.utils import (
    set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer,
)
from utils.direct_neighbor_sampler import get_neighbor_sampler
from utils.utils import get_neighbor_sampler as get_undirected_sampler
from utils.utils import NegativeEdgeSampler
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_sign_prediction_args

TASK_NAME = "DirectLinkPred"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _build_edge_index(batch_src, batch_dst):
    """构建 (src, dst) → indices 映射，用于批次内反向边检测。"""
    index = {}
    for i, (s, d) in enumerate(zip(batch_src, batch_dst)):
        key = (int(s), int(d))
        index.setdefault(key, []).append(i)
    return index


def _extract_reverse(
    batch_src, batch_dst, batch_times, edge_index,
):
    """
    从批次中提取反向边样本。
    :return: rev_src, rev_dst, rev_times — 三个 ndarray，可能为空
    """
    rev_src_list, rev_dst_list, rev_time_list = [], [], []
    seen = set()
    for i, (s, d) in enumerate(zip(batch_src, batch_dst)):
        key = (int(s), int(d))
        rev_key = (int(d), int(s))
        if rev_key in edge_index and key not in seen:
            seen.add(key)
            seen.add(rev_key)
            rev_src_list.append(d)
            rev_dst_list.append(s)
            rev_time_list.append(batch_times[i])
    return (
        np.array(rev_src_list, dtype=np.longlong),
        np.array(rev_dst_list, dtype=np.longlong),
        np.array(rev_time_list, dtype=np.float32),
    )


def _get_sampler(model_name, directed_sampler, undirected_sampler):
    """根据模型返回对应的邻居采样器。"""
    if model_name in ["TGAT", "GraphMixer"]:
        return undirected_sampler
    return directed_sampler


def _compute_embeddings(model, src, dst, times, sign, model_name, num_neighbors, time_gap):
    """模型前向 dispatch。"""
    if model_name in ["TGAT"]:
        return model.compute_src_dst_node_temporal_embeddings(
            src, dst, times, num_neighbors=num_neighbors)
    elif model_name in ["GraphMixer"]:
        return model.compute_src_dst_node_temporal_embeddings(
            src, dst, times, num_neighbors=num_neighbors, time_gap=time_gap)
    elif model_name in [DyGFormer.NAME]:
        return model.compute_src_dst_node_temporal_embeddings(src, dst, times)
    elif model_name in [SignDyGFormer.NAME, DirectSignDyGFormer.NAME]:
        return model.compute_src_dst_node_temporal_embeddings(src, dst, times, sign)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = get_sign_prediction_args(is_evaluation=False)

    (node_raw_features, edge_raw_features, full_data,
     train_data, val_data, test_data,
     new_node_val_data, new_node_test_data,
     ) = get_link_prediction_data(
        dataset_name=args.dataset_name,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        tail_num=args.tail_num,
    )

    # ---- 采样器 ----
    train_neighbor_sampler = get_neighbor_sampler(
        data=train_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=0,
        common_neighbor_look_forward=args.common_neighbors_look_forward,
        module_repeat_aware_sampler=args.module_repeat_aware_sampler,
        module_common_neighbor_sampler=args.module_common_neighbor_aware_sampler,
    )
    full_neighbor_sampler = get_neighbor_sampler(
        data=full_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=1,
        common_neighbor_look_forward=args.common_neighbors_look_forward,
        module_repeat_aware_sampler=args.module_repeat_aware_sampler,
        module_common_neighbor_sampler=args.module_common_neighbor_aware_sampler,
    )

    # ---- 无向采样器 (TGAT / GraphMixer 用) ----
    train_undirected_sampler = get_undirected_sampler(
        data=train_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=0,
    )
    full_undirected_sampler = get_undirected_sampler(
        data=full_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=1,
    )

    # ---- 负采样器 ----
    train_neg_sampler = NegativeEdgeSampler(
        src_node_ids=train_data.src_node_ids,
        dst_node_ids=train_data.dst_node_ids,
    )

    # ---- DataLoaders ----
    train_loader = get_idx_data_loader(
        indices_list=list(range(len(train_data.src_node_ids))),
        batch_size=args.batch_size, shuffle=False,
    )
    val_loader = get_idx_data_loader(
        indices_list=list(range(len(val_data.src_node_ids))),
        batch_size=args.batch_size, shuffle=False,
    )
    test_loader = get_idx_data_loader(
        indices_list=list(range(len(test_data.src_node_ids))),
        batch_size=args.batch_size, shuffle=False,
    )

    # ---- 多 run ----
    test_metric_all_runs = []

    for run in range(args.num_runs):
        set_random_seed(seed=args.seeds[run] if run < len(args.seeds) else run)
        args.seed = args.seeds[run] if run < len(args.seeds) else run
        args.save_model_name = f"{args.model_name}_seed{args.seed}"

        # logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        log_dir = f"./logs/{TASK_NAME}/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(f"{log_dir}{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler(); ch.setLevel(logging.WARNING)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt); ch.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(ch)

        logger.info(f"********** Run {run+1} **********")
        logger.info(f"config: {args}")

        # ---- 模型 ----
        if args.model_name == "TGAT":
            dynamic_backbone = TGAT(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=train_undirected_sampler,
                time_feat_dim=args.time_feat_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                device=args.device,
            )
        elif args.model_name == "GraphMixer":
            dynamic_backbone = GraphMixer(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=train_undirected_sampler,
                time_feat_dim=args.time_feat_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                device=args.device,
            )
        elif args.model_name == DyGFormer.NAME:
            dynamic_backbone = DyGFormer(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                patch_size=args.patch_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=args.device,
            )
        elif args.model_name == SignDyGFormer.NAME:
            dynamic_backbone = SignDyGFormer(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                patch_size=args.patch_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=args.device,
            )
        elif args.model_name == DirectSignDyGFormer.NAME:
            dynamic_backbone = DirectSignDyGFormer(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                patch_size=args.patch_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=args.device,
                module_status_theory_encoder=args.module_status_theory_encoder,
                module_balance_fallback=args.module_balance_fallback,
            )
        else:
            raise ValueError(f"Unknown model: {args.model_name}")
        link_predictor = MergeLayer(
            input_dim1=node_raw_features.shape[1],
            input_dim2=node_raw_features.shape[1],
            hidden_dim=node_raw_features.shape[1],
            output_dim=1,
        )
        model = nn.Sequential(dynamic_backbone, link_predictor)
        model = convert_to_gpu(model, device=args.device)
        logger.info(f"#params: {get_parameter_sizes(model)*4/1024:.1f} KB")

        optimizer = create_optimizer(model, args.optimizer, args.learning_rate, args.weight_decay)

        save_dir = f"./saved_models/{TASK_NAME}/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        os.makedirs(save_dir, exist_ok=True)
        # 2026-09-14 修复（并发互删缺陷）：原为整目录 shutil.rmtree(save_dir)——
        # 同数据集同 seed 的并发任务会互删对方 best 模型 → 结尾 load_checkpoint FileNotFound。
        # 现仅清理本次 run 的目标文件（防误载同名旧文件；串行语义与原来一致）。
        for _stale_name in (
            f"{args.save_model_name}.pkl",
            f"{args.save_model_name}.param.json",
            f"{args.save_model_name}_nonparametric_data.pkl",
        ):
            _stale_path = os.path.join(save_dir, _stale_name)
            if os.path.exists(_stale_path):
                os.remove(_stale_path)

        early_stopping = EarlyStopping(
            patience=args.patience, save_model_folder=save_dir,
            save_model_name=args.save_model_name, logger=logger, model_name=args.model_name,
        )

        link_loss = LinkLoss(LinkLossType.BCE, EdgeDirectionMode.DIRECTED)

        # ---- 训练 ----
        for epoch in range(args.num_epochs):
            model.train()
            model[0].set_neighbor_sampler(
                _get_sampler(args.model_name, train_neighbor_sampler, train_undirected_sampler)
            )

            train_losses, train_metrics = [], []
            for batch_indices in tqdm(train_loader, ncols=120, desc=f"Epoch {epoch+1}"):
                idx = batch_indices.numpy()
                batch_src = train_data.src_node_ids[idx]
                batch_dst = train_data.dst_node_ids[idx]
                batch_times = train_data.node_interact_times[idx]
                batch_sign = train_data.node_interact_sign[idx]

                # ---- pos 样本 ----
                pos_src_emb, pos_dst_emb = _compute_embeddings(
                    model[0], batch_src, batch_dst, batch_times, batch_sign,
                    args.model_name, args.num_neighbors, args.time_gap,
                )
                pos_logits = model[1](input_1=pos_src_emb, input_2=pos_dst_emb)

                # ---- rev 样本 (批次内反向边) ----
                edge_idx = _build_edge_index(batch_src, batch_dst)
                rev_src, rev_dst, rev_times = _extract_reverse(
                    batch_src, batch_dst, batch_times, edge_idx,
                )
                if len(rev_src) > 0:
                    rev_src_emb, rev_dst_emb = _compute_embeddings(
                        model[0], rev_src, rev_dst, rev_times,
                        np.zeros(len(rev_src), dtype=np.int8),
                        args.model_name, args.num_neighbors, args.time_gap,
                    )
                    rev_logits = model[1](input_1=rev_src_emb, input_2=rev_dst_emb)
                else:
                    rev_logits = None

                # ---- neg 样本 (随机负采样, N_neg = N_rev; 无反向时 1:1) ----
                n_neg = len(rev_src) if len(rev_src) > 0 else len(batch_src)
                _, neg_dst = train_neg_sampler.sample(size=n_neg)
                neg_src = batch_src[:n_neg]  # keep src, replace dst

                neg_src_emb, neg_dst_emb = _compute_embeddings(
                    model[0], neg_src, neg_dst, batch_times[:n_neg],
                    np.zeros(n_neg, dtype=np.int8),
                    args.model_name, args.num_neighbors, args.time_gap,
                )
                neg_logits = model[1](input_1=neg_src_emb, input_2=neg_dst_emb)

                # ---- loss + metrics ----
                loss, batch_logits, batch_labels = link_loss(
                    pos_logits=pos_logits,
                    neg_logits=neg_logits,
                    rev_logits=rev_logits,
                    return_logits=True,
                )

                train_losses.append(loss.item())
                train_metrics.append(
                    get_link_prediction_metrics(
                        predicts=batch_logits, labels=batch_labels,
                    )
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # ---- epoch 汇总 ----
            mean_loss = np.mean(train_losses)
            mean_metrics = {
                k: np.mean([m[k] for m in train_metrics]) for k in train_metrics[0]
            }
            logger.info(
                f"Epoch {epoch+1}: train_loss={mean_loss:.4f} | "
                + " ".join(f"{k}={v:.4f}" for k, v in mean_metrics.items())
            )

            # ---- 验证 + 早停 ----
            val_losses, val_metrics, best_thr = evaluate_directed_link_prediction(
                model=model,
                neighbor_sampler=_get_sampler(args.model_name, full_neighbor_sampler, full_undirected_sampler),
                evaluate_idx_data_loader=val_loader,
                evaluate_data=val_data,
                loss_fn=link_loss,
                neg_sampler=train_neg_sampler,
                model_name=args.model_name,
                num_neighbors=args.num_neighbors,
                time_gap=args.time_gap,
                device=args.device,
            )
            val_metrics_dict = val_metrics[0]
            logger.info(
                f"Epoch {epoch+1}: val_loss={np.mean(val_losses):.4f} | "
                + " ".join(f"{k}={v:.4f}" for k, v in val_metrics_dict.items())
                + f" | best_thr={best_thr:.4f}"
            )

            val_metric_indicator = [
                (name, val_metrics_dict[name], True)
                for name in val_metrics_dict
            ]
            if early_stopping.step(
                val_metric_indicator, model, hyper_parm={"thr": float(best_thr)}
            ):
                logger.info("Early stopping triggered.")
                break

            # ---- 测试 (定期) ----
            if (epoch + 1) % args.test_interval_epochs == 0:
                test_losses, test_metrics, _ = evaluate_directed_link_prediction(
                    model=model,
                    neighbor_sampler=_get_sampler(args.model_name, full_neighbor_sampler, full_undirected_sampler),
                    evaluate_idx_data_loader=test_loader,
                    evaluate_data=test_data,
                    loss_fn=link_loss,
                    neg_sampler=train_neg_sampler,
                    model_name=args.model_name,
                    num_neighbors=args.num_neighbors,
                    time_gap=args.time_gap,
                    thr=best_thr,
                    device=args.device,
                )
                logger.info(
                    f"Epoch {epoch+1}: test_loss={np.mean(test_losses):.4f} | "
                    + " ".join(f"{k}={v:.4f}" for k, v in test_metrics[0].items())
                )

        # ---- 加载最优模型 → 最终测试 ----
        early_stopping.load_checkpoint(model)
        hyper_param = early_stopping.load_hyper_param()
        final_thr = hyper_param.get("thr", 0.5) if hyper_param else 0.5

        logger.info(f"Final evaluation with best_thr={final_thr:.4f} ...")
        test_losses, test_metrics, _ = evaluate_directed_link_prediction(
            model=model,
            neighbor_sampler=_get_sampler(args.model_name, full_neighbor_sampler, full_undirected_sampler),
            evaluate_idx_data_loader=test_loader,
            evaluate_data=test_data,
            loss_fn=link_loss,
            neg_sampler=train_neg_sampler,
            model_name=args.model_name,
            num_neighbors=args.num_neighbors,
            time_gap=args.time_gap,
            thr=final_thr,
            device=args.device,
        )
        test_metrics_dict = test_metrics[0]
        for k, v in test_metrics_dict.items():
            logger.info(f"  test {k}: {v:.4f}")
        test_metric_all_runs.append(test_metrics_dict)

        # ---- 保存结果 ----
        save_result_folder = f"./saved_results/{TASK_NAME}/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        result_path = os.path.join(save_result_folder, f"{args.result_save_name}.json")
        with open(result_path, "w") as f:
            json.dump(
                {k: f"{v:.4f}" for k, v in test_metrics_dict.items()}, f, indent=4,
            )

        logger.info(f"Run {run+1} finished.")

    # ---- 多 run 汇总 ----
    logger.info(f"=== Summary over {len(test_metric_all_runs)} runs ===")
    for metric_name in test_metric_all_runs[0]:
        vals = [r[metric_name] for r in test_metric_all_runs]
        logger.info(f"  test {metric_name}: {np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}")
