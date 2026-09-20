"""
DirectSignDyGFormer 有向符号链路预测训练脚本
基于 train_link_sign_prediction.py，新增 DirectSignDyGFormer 支持。
"""
import datetime
import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import json
import torch
import torch.nn as nn

from models.DirectSignDyGFormer import DirectSignDyGFormer
from models.DyGFormer import DyGFormer
from models.modules import MergeLayer
from utils.metrics.signPredict import get_sign_prediction_metrics
from utils.utils import (
    set_random_seed,
    convert_to_gpu,
    get_parameter_sizes,
    create_optimizer,
)
from utils.direct_neighbor_sampler import get_neighbor_sampler
from utils.utils import NegativeEdgeSampler
from evaluate_models_utils import evaluate_model_sign_prediction
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_sign_prediction_args
from loss_function import FocalLoss

TASK_NAME = "LinkSign"

# 早停判据键集合（2026-09-20 用户批准修复）：恢复 09-18 指标扩展前的历史口径——
# best-checkpoint/早停判据 = 这些键"全部同时不下降"（patience 计数）；扩展键
# （precision/recall/balanced_acc/mcc/thr 等）仍照常写入结果 JSON，但不参与判据。
EARLY_STOP_METRICS = ["ap", "f1_macro", "f1_binary", "acc", "auc", "f1_weighted"]

if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    args = get_sign_prediction_args(is_evaluation=False)

    # ---- 加载数据 ----
    (
        node_raw_features,
        edge_raw_features,
        full_data,
        train_data,
        val_data,
        test_data,
        new_node_val_data,
        new_node_test_data,
    ) = get_link_prediction_data(
        dataset_name=args.dataset_name,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        tail_num=args.tail_num,
    )

    # ---- 初始化采样器 ----
    train_neighbor_sampler = get_neighbor_sampler(
        data=train_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=0,
        common_neighbor_look_forward=args.common_neighbors_look_forward,
        ras_look_forward=args.ras_look_forward,
        module_repeat_aware_sampler=args.module_repeat_aware_sampler,
        module_common_neighbor_sampler=args.module_common_neighbor_aware_sampler,
    )
    full_neighbor_sampler = get_neighbor_sampler(
        data=full_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=1,
        common_neighbor_look_forward=args.common_neighbors_look_forward,
        ras_look_forward=args.ras_look_forward,
        module_repeat_aware_sampler=args.module_repeat_aware_sampler,
        module_common_neighbor_sampler=args.module_common_neighbor_aware_sampler,
    )

    # ---- 负采样器 ----
    train_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids
    )
    val_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=full_data.src_node_ids,
        dst_node_ids=full_data.dst_node_ids,
        seed=0,
    )
    new_node_val_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=new_node_val_data.src_node_ids,
        dst_node_ids=new_node_val_data.dst_node_ids,
        seed=1,
    )
    test_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=full_data.src_node_ids,
        dst_node_ids=full_data.dst_node_ids,
        seed=2,
    )
    new_node_test_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=new_node_test_data.src_node_ids,
        dst_node_ids=new_node_test_data.dst_node_ids,
        seed=3,
    )

    # ---- DataLoaders ----
    train_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(train_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )
    val_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(val_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )
    new_node_val_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(new_node_val_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(test_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )
    new_node_test_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(new_node_test_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # ---- 多 run 训练 ----
    (
        val_metric_all_runs,
        new_node_val_metric_all_runs,
        test_metric_all_runs,
        new_node_test_metric_all_runs,
    ) = ([], [], [], [])

    for run in range(args.num_runs):
        set_random_seed(seed=args.seeds[run] if run < len(args.seeds) else run)
        args.seed = args.seeds[run] if run < len(args.seeds) else run
        args.save_model_name = f"{args.model_name}_seed{args.seed}"

        # ---- Logger ----
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        log_dir = f"./logs/{TASK_NAME}/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(f"{log_dir}{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")
        logger.info(f"configuration is {args}")

        # ---- 创建模型 ----
        if args.model_name == "DyGFormer":
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
        elif args.model_name == "DirectSignDyGFormer":
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
            raise ValueError(f"Wrong value for model_name {args.model_name}!")

        link_predictor = MergeLayer(
            input_dim1=node_raw_features.shape[1],
            input_dim2=node_raw_features.shape[1],
            hidden_dim=node_raw_features.shape[1],
            output_dim=1,
        )
        model = nn.Sequential(dynamic_backbone, link_predictor)
        logger.info(f"model -> {model}")
        logger.info(
            f"model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B"
        )

        optimizer = create_optimizer(
            model=model,
            optimizer_name=args.optimizer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        model = convert_to_gpu(model, device=args.device)

        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        os.makedirs(save_model_folder, exist_ok=True)
        # 2026-09-14 修复（并发互删缺陷）：原为整目录 shutil.rmtree(save_model_folder)——
        # 同数据集同 seed 的并发任务会互删对方 best 模型 → 结尾 load_checkpoint FileNotFound。
        # 现仅清理本次 run 的目标文件（防误载同名旧文件；串行语义与原来一致）。
        for _stale_name in (
            f"{args.save_model_name}.pkl",
            f"{args.save_model_name}.param.json",
            f"{args.save_model_name}_nonparametric_data.pkl",
        ):
            _stale_path = os.path.join(save_model_folder, _stale_name)
            if os.path.exists(_stale_path):
                os.remove(_stale_path)

        early_stopping = EarlyStopping(
            patience=args.patience,
            save_model_folder=save_model_folder,
            save_model_name=args.save_model_name,
            logger=logger,
            model_name=args.model_name,
            metric_notice=EARLY_STOP_METRICS,
        )

        loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight]).to(args.device))

        # ---- 训练循环 ----
        for epoch in range(args.num_epochs):
            model.train()
            if args.model_name in ["DyGFormer", "DirectSignDyGFormer"]:
                model[0].set_neighbor_sampler(train_neighbor_sampler)

            train_losses, train_metrics = [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                (
                    batch_src_node_ids,
                    batch_dst_node_ids,
                    batch_node_interact_times,
                    batch_edge_ids,
                    batch_sign,
                ) = (
                    train_data.src_node_ids[train_data_indices],
                    train_data.dst_node_ids[train_data_indices],
                    train_data.node_interact_times[train_data_indices],
                    train_data.edge_ids[train_data_indices],
                    train_data.node_interact_sign[train_data_indices],
                )

                # 过滤中立边 (sign == 0)
                mask = batch_sign.squeeze() != 0
                batch_src_node_ids = batch_src_node_ids[mask]
                batch_dst_node_ids = batch_dst_node_ids[mask]
                batch_node_interact_times = batch_node_interact_times[mask]
                batch_edge_ids = batch_edge_ids[mask]
                batch_sign = batch_sign[mask]

                # ---- 前向传播 ----
                if args.model_name in ["DyGFormer"]:
                    batch_src_node_embeddings, batch_dst_node_embeddings = model[
                        0
                    ].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                    )
                elif args.model_name in ["DirectSignDyGFormer"]:
                    batch_src_node_embeddings, batch_dst_node_embeddings = model[
                        0
                    ].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        node_interact_sign=batch_sign,
                    )
                else:
                    raise ValueError(f"Wrong value for model_name {args.model_name}!")

                # 链路预测
                positive_probabilities = model[1](
                    input_1=batch_src_node_embeddings,
                    input_2=batch_dst_node_embeddings,
                ).squeeze(-1)

                positive_probabilities_filter = positive_probabilities
                negative_probabilities = positive_probabilities_filter[batch_sign == -1]
                positive_probabilities = positive_probabilities_filter[batch_sign == 1]

                predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                labels = torch.cat(
                    [
                        torch.ones(positive_probabilities.size(0), device=positive_probabilities.device, dtype=torch.float),
                        torch.zeros(negative_probabilities.size(0), device=negative_probabilities.device, dtype=torch.float),
                    ],
                )

                loss = loss_func.forward(predicts, labels)
                train_losses.append(loss.item())
                train_metrics.append(get_sign_prediction_metrics(predicts=predicts, labels=labels))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_idx_data_loader_tqdm.set_description(
                    f"Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}"
                )

            # ---- 验证 ----
            val_losses, val_metrics, best_thr = evaluate_model_sign_prediction(
                model_name=args.model_name,
                model=model,
                neighbor_sampler=full_neighbor_sampler,
                evaluate_idx_data_loader=val_idx_data_loader,
                evaluate_data=val_data,
                loss_func=loss_func,
                num_neighbors=args.num_neighbors,
                time_gap=args.time_gap,
            )
            new_node_val_losses, new_node_val_metrics, _ = evaluate_model_sign_prediction(
                model_name=args.model_name,
                model=model,
                neighbor_sampler=full_neighbor_sampler,
                evaluate_idx_data_loader=new_node_val_idx_data_loader,
                evaluate_data=new_node_val_data,
                loss_func=loss_func,
                num_neighbors=args.num_neighbors,
                time_gap=args.time_gap,
                thr=best_thr,
            )

            logger.info(
                f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}'
            )
            for metric_name in train_metrics[0].keys():
                logger.info(
                    f"train {metric_name}, {np.mean([m[metric_name] for m in train_metrics]):.4f}"
                )
            logger.info(f"validate loss: {np.mean(val_losses):.4f}")
            for metric_name in val_metrics[0].keys():
                logger.info(
                    f"validate {metric_name}, {np.mean([m[metric_name] for m in val_metrics]):.4f}"
                )

            # ---- 早停 + 测试 ----
            if (epoch + 1) % args.test_interval_epochs == 0:
                test_losses, test_metrics, _ = evaluate_model_sign_prediction(
                    model_name=args.model_name,
                    model=model,
                    neighbor_sampler=full_neighbor_sampler,
                    evaluate_idx_data_loader=test_idx_data_loader,
                    evaluate_data=test_data,
                    loss_func=loss_func,
                    num_neighbors=args.num_neighbors,
                    time_gap=args.time_gap,
                    thr=best_thr,
                )
                new_node_test_losses, new_node_test_metrics, _ = evaluate_model_sign_prediction(
                    model_name=args.model_name,
                    model=model,
                    neighbor_sampler=full_neighbor_sampler,
                    evaluate_idx_data_loader=new_node_test_idx_data_loader,
                    evaluate_data=new_node_test_data,
                    loss_func=loss_func,
                    num_neighbors=args.num_neighbors,
                    time_gap=args.time_gap,
                    thr=best_thr,
                )
                logger.info(f"test loss: {np.mean(test_losses):.4f}")
                for metric_name in test_metrics[0].keys():
                    logger.info(
                        f"test {metric_name}, {np.mean([m[metric_name] for m in test_metrics]):.4f}"
                    )

            early_stopping.update(val_losses, model)

        # ---- 记录 run 结果 ----
        val_metric_all_runs.append(val_metrics)
        new_node_val_metric_all_runs.append(new_node_val_metrics)
        test_metric_all_runs.append(test_metrics)
        new_node_test_metric_all_runs.append(new_node_test_metrics)

        logger.info(f"run {run + 1} cost {time.time() - run_start_time:.2f} s")

    # ---- 汇总输出 ----
    logger.info("********** All runs finished **********")
    for metric_name in test_metric_all_runs[0][0].keys():
        means = [np.mean([m[metric_name] for m in metrics]) for metrics in test_metric_all_runs]
        logger.info(f"test {metric_name} across runs: {np.mean(means):.4f} ± {np.std(means):.4f}")
