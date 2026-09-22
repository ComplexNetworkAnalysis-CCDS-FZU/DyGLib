from datetime import datetime
import logging
import time
import sys
import os
from sklearn.utils import compute_class_weight
from tqdm import tqdm
import numpy as np
import warnings
import json
import torch
import torch.nn as nn

from models.SignDyGFormer import SignDyGFormer
from models.DyGFormer import DyGFormer
from models.modules import SignNullClassifyLayer, cascade_loss, sign_link3class_label
from utils.metrics.linkSignPredict import get_linksign_prediction_metrics
from utils.utils import (
    dataset_sampler,
    set_random_seed,
    convert_to_gpu,
    get_parameter_sizes,
    create_optimizer,
)
from utils.direct_neighbor_sampler import get_neighbor_sampler
from utils.utils import NegativeEdgeSampler
from evaluate_models_utils import evaluate_model_sign_link_3class_prediction
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_sign_prediction_args
from utils import accel as _accel

TASK_NAME= "SignLinkPrediction"

# 早停判据键集合（2026-09-20 用户批准修复）：恢复 09-18 指标扩展前的历史口径——
# best-checkpoint/早停判据 = 这些键"全部同时不下降"（patience 计数）；扩展键
# （auc_wt/precision_neg/recall_neg/precision_pos/recall_pos/mcc/thr_exist/thr_sign）
# 仍照常写入结果 JSON，但不参与判据，保证与 09-18 前批次（主表/LOO 早期等）同口径。
EARLY_STOP_METRICS = [
    "exist_recall", "exist_precision", "exist_f1", "sign_f1", "ap",
    "f1_mac", "f1_wt", "f1_mic", "acc", "auc",
]

if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    # get arguments
    args = get_sign_prediction_args(is_evaluation=False)

    # get data for training, validation and testing
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
        noise_ratio=args.noise_ratio,
        noise_seed=args.noise_seed,
        noise_scope=args.noise_scope,
    )

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(
        data=train_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=0,
        common_neighbor_look_forward=args.common_neighbors_look_forward,
        ras_look_forward=args.ras_look_forward,
        module_repeat_aware_sampler=args.module_repeat_aware_sampler,
        module_common_neighbor_sampler=args.module_common_neighbor_aware_sampler,
        cnas_tail_fill=args.cnas_tail_fill,
        recent_block=args.recent_block,
    )

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(
        data=full_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=1,
        common_neighbor_look_forward=args.common_neighbors_look_forward,
        ras_look_forward=args.ras_look_forward,
        module_repeat_aware_sampler=args.module_repeat_aware_sampler,
        module_common_neighbor_sampler=args.module_common_neighbor_aware_sampler,
        cnas_tail_fill=args.cnas_tail_fill,
        recent_block=args.recent_block,
    )

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    # train negative edge sampler does not need to specify the seed, but evaluation samplers need to do so
    train_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids
    )
    val_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=0
    )
    new_node_val_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=new_node_val_data.src_node_ids,
        dst_node_ids=new_node_val_data.dst_node_ids,
        seed=1,
    )
    test_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2
    )
    new_node_test_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=new_node_test_data.src_node_ids,
        dst_node_ids=new_node_test_data.dst_node_ids,
        seed=3,
    )

    # 计算正负边占比
    classes = np.array([1, -1])
    non_zero = train_data.node_interact_sign[train_data.node_interact_sign != 0]
    weights = compute_class_weight("balanced", classes=np.unique(non_zero), y=non_zero)
    weights = np.array([weights[0], weights[1], 1], dtype=np.float32)

    print(f"positive, negative, null sample weight are {weights}")

    sampler = dataset_sampler(train_data)

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(train_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
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

    (
        val_metric_all_runs,
        new_node_val_metric_all_runs,
        test_metric_all_runs,
        new_node_test_metric_all_runs,
    ) = ([], [], [], [])

    for run,run_seed in enumerate(args.seeds):

        set_random_seed(seed=run_seed)

        args.seed = run_seed
        args.save_model_name = f"{args.model_name}_seed{args.seed}"

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(
            f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/",
            exist_ok=True,
        )
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(
            f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log"
        )
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        # E-1: 每次 run 重置峰值显存统计，确保峰值反映本次 run
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f"configuration is {args}")

        # create model
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
        elif args.model_name == "SignDyGFormer":
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
                module_repeat_aware_sign_encoder=args.module_repeat_aware_sign_encoder,
                module_balance_theory_encoder=args.module_balance_theory_encoder,
                module_common_neighbor_encoder=args.module_common_neighbor_encoder,
                module_balance_theory_gate=args.module_balance_theory_gate,
                time_decay_lambda=args.time_decay_lambda,
                time_decay_gap_mode=args.time_decay_gap_mode,
                time_scaling_factor=args.time_scaling_factor,
            )
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")
        # 符号链路预测任务，这里变为3分类问题：链路可能情况： 0：正，1：负，2：中立连接（0）,3：无连接（其他）
        link_sign_predictor = SignNullClassifyLayer(
            input_dim1=node_raw_features.shape[1],
            input_dim2=node_raw_features.shape[1],
            null_input_dim1=node_raw_features.shape[1],
            null_input_dim2=node_raw_features.shape[1],
            hidden_dim=node_raw_features.shape[1],
            reject_support=args.reject_support,
        )
        model = nn.Sequential(dynamic_backbone, link_sign_predictor)
        logger.info(f"model -> {model}")
        logger.info(
            f"model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, "
            f"{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB."
        )

        optimizer = create_optimizer(
            model=model,
            optimizer_name=args.optimizer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        model = convert_to_gpu(model, device=args.device)

        save_model_folder = f"./saved_models/{TASK_NAME}/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        os.makedirs(save_model_folder, exist_ok=True)
        # 2026-09-14 修复（并发互删缺陷）：原为整目录 shutil.rmtree(save_model_folder)——
        # 同数据集同 seed 的并发任务会互删对方 best 模型 → 结尾 load_checkpoint FileNotFound。
        # 现仅清理本次 run 的目标文件（防误载同名旧文件；串行语义与原来一致）。
        for _stale_name in (
            f"{args.result_save_name}.pkl",
            f"{args.result_save_name}.param.json",
            f"{args.result_save_name}_nonparametric_data.pkl",
        ):
            _stale_path = os.path.join(save_model_folder, _stale_name)
            if os.path.exists(_stale_path):
                os.remove(_stale_path)

        early_stopping = EarlyStopping(
            patience=args.patience,
            save_model_folder=save_model_folder,
            save_model_name=args.result_save_name,
            logger=logger,
            model_name=args.model_name,
            metric_notice=EARLY_STOP_METRICS,
        )

        loss_func = nn.BCEWithLogitsLoss()

        for epoch in range(args.num_epochs):

            model.train()
            if args.model_name in [
                "DyGFormer",
                "SignDyGFormer",
            ]:
                # training, only use training graph
                model[0].set_neighbor_sampler(train_neighbor_sampler)

            # store train losses and metrics
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

                mask = batch_sign.squeeze() != 0

                batch_src_node_ids = batch_src_node_ids[mask]
                batch_dst_node_ids = batch_dst_node_ids[mask]
                batch_node_interact_times = batch_node_interact_times[mask]
                batch_edge_ids = batch_edge_ids[mask]
                batch_sign = batch_sign[mask]

                _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(
                    size=len(batch_src_node_ids)
                )
                batch_neg_src_node_ids = batch_src_node_ids

                # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
                # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes

                if args.model_name in ["DyGFormer"]:
                    # get temporal embedding of source and destination nodes
                    batch_src_node_embeddings, batch_dst_node_embeddings = model[
                        0
                    ].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                    )

                    # get temporal embedding of negative source and negative destination nodes
                    (
                        batch_neg_src_node_embeddings,
                        batch_neg_dst_node_embeddings,
                    ) = model[0].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_neg_src_node_ids,
                        dst_node_ids=batch_neg_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                    )
                elif args.model_name in ["SignDyGFormer"]:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = model[
                        0
                    ].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        node_interact_sign=batch_sign,
                    )

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    # 负样本的符号呢？
                    # - 可能需要考虑平衡理论，这里先简单处理，全部为0
                    (
                        batch_neg_src_node_embeddings,
                        batch_neg_dst_node_embeddings,
                    ) = model[0].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_neg_src_node_ids,
                        dst_node_ids=batch_neg_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        node_interact_sign=np.zeros_like(batch_sign),
                    )
                else:
                    raise ValueError(f"Wrong value for model_name {args.model_name}!")
                # get positive and negative probabilities, shape (batch_size, )
                exist_predict, exist_prob, sign_predict = model[1](
                    input_1=batch_src_node_embeddings,
                    input_2=batch_dst_node_embeddings,
                    null_input_1=batch_neg_src_node_embeddings,
                    null_input_2=batch_neg_dst_node_embeddings,
                )

                y_exist, y_sign = sign_link3class_label(
                    src_emb=batch_src_node_embeddings,
                    dst_emb=batch_dst_node_embeddings,
                    neg_src_emb=batch_neg_src_node_embeddings,
                    neg_dst_emb=batch_neg_dst_node_embeddings,
                    edge_sign=batch_sign,
                    reject_support=args.reject_support,
                )

                loss = cascade_loss(
                    exist_predict,
                    sign_predict,
                    exist_prob,
                    y_exist,
                    y_sign,
                    reject_support=args.reject_support,
                )
                train_losses.append(loss.item())

                train_metrics.append(
                    get_linksign_prediction_metrics(
                        sign_predicts=sign_predict,
                        sign_labels=y_sign,
                        exist_predicts=exist_predict,
                        exist_labels=y_exist,
                        reject_support=args.reject_support,
                    )
                )

                optimizer.zero_grad()
                loss.backward()

                # g = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # print(f'grad norm={g:.4f} lr={optimizer.param_groups[0]["lr"]}')

                optimizer.step()

                train_idx_data_loader_tqdm.set_description(
                    f"Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item():4f}"
                )
                # total_norm = 0

                # for name, p in model.named_parameters():
                #     if p.grad is not None:
                #         param_norm = p.grad.data.norm(2).item()
                #         total_norm += param_norm**2
                #         if param_norm < 1e-7:
                #             print(name, param_norm)  # 谁最先变成 0
                #     total_norm = total_norm**0.5
                # print("total_grad_norm", total_norm)

                # g = (
                #     model[0]
                #     .neighbor_co_occurrence_encoder.neighbor_sign_effect_layer[0]
                #     .weight.grad.norm()
                #     .item()
                # )
                # print("sparse linear grad:", g)

            val_losses, val_metrics, best_exist_thr, best_sign_thr = (
                evaluate_model_sign_link_3class_prediction(
                    model_name=args.model_name,
                    model=model,
                    neighbor_sampler=full_neighbor_sampler,
                    evaluate_idx_data_loader=val_idx_data_loader,
                    evaluate_neg_edge_sampler=val_neg_edge_sampler,
                    evaluate_data=val_data,
                    loss_func=loss_func,
                    num_neighbors=args.num_neighbors,
                    time_gap=args.time_gap,
                    exist_best_thr=args.exist_thr,
                    sign_best_thr=args.sign_thr,
                )
            )

            new_node_val_losses, new_node_val_metrics, best_exist_thr, best_sign_thr = (
                evaluate_model_sign_link_3class_prediction(
                    model_name=args.model_name,
                    model=model,
                    neighbor_sampler=full_neighbor_sampler,
                    evaluate_idx_data_loader=new_node_val_idx_data_loader,
                    evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                    evaluate_data=new_node_val_data,
                    loss_func=loss_func,
                    num_neighbors=args.num_neighbors,
                    time_gap=args.time_gap,
                    sign_best_thr=best_sign_thr,
                    exist_best_thr=best_exist_thr,
                )
            )

            logger.info(
                f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}'
            )
            for metric_name in train_metrics[0].keys():
                logger.info(
                    f"train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}"
                )
            logger.info(f"validate loss: {np.mean(val_losses):.4f}")
            for metric_name in val_metrics[0].keys():
                logger.info(
                    f"validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}"
                )
            logger.info(f"new node validate loss: {np.mean(new_node_val_losses):.4f}")
            for metric_name in new_node_val_metrics[0].keys():
                logger.info(
                    f"new node validate {metric_name}, {np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics]):.4f}"
                )

            # perform testing once after test_interval_epochs
            if (epoch + 1) % args.test_interval_epochs == 0:
                model[0].profiler.enable()  # 开启模型内部的Profiler以记录测试阶段的时间
                test_losses, test_metrics, best_exist_thr, best_sign_thr = (
                    evaluate_model_sign_link_3class_prediction(
                        model_name=args.model_name,
                        model=model,
                        neighbor_sampler=full_neighbor_sampler,
                        evaluate_idx_data_loader=test_idx_data_loader,
                        evaluate_neg_edge_sampler=test_neg_edge_sampler,
                        evaluate_data=test_data,
                        loss_func=loss_func,
                        num_neighbors=args.num_neighbors,
                        time_gap=args.time_gap,
                        sign_best_thr=best_sign_thr,
                        exist_best_thr=best_exist_thr,
                    )
                )

                (
                    new_node_test_losses,
                    new_node_test_metrics,
                    best_exist_thr,
                    best_sign_thr,
                ) = evaluate_model_sign_link_3class_prediction(
                    model_name=args.model_name,
                    model=model,
                    neighbor_sampler=full_neighbor_sampler,
                    evaluate_idx_data_loader=new_node_test_idx_data_loader,
                    evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                    evaluate_data=new_node_test_data,
                    loss_func=loss_func,
                    num_neighbors=args.num_neighbors,
                    time_gap=args.time_gap,
                    sign_best_thr=best_sign_thr,
                    exist_best_thr=best_exist_thr,
                )
                model[0].profiler.disable()  # 关闭模型内部的Profiler

                logger.info(f"test loss: {np.mean(test_losses):.4f}")
                for metric_name in test_metrics[0].keys():
                    logger.info(
                        f"test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}"
                    )
                logger.info(f"new node test loss: {np.mean(new_node_test_losses):.4f}")
                for metric_name in new_node_test_metrics[0].keys():
                    logger.info(
                        f"new node test {metric_name}, {np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics]):.4f}"
                    )
                model[0].profiler.summary()  # 输出测试阶段的时间汇总统计

            # select the best model based on all the validate metrics
            val_metric_indicator = []
            for metric_name in val_metrics[0].keys():
                val_metric_indicator.append(
                    (
                        metric_name,
                        np.mean(
                            [val_metric[metric_name] for val_metric in val_metrics]
                        ),
                        True,
                    )
                )
            early_stop = early_stopping.step(
                val_metric_indicator,
                model,
                hyper_parm={
                    "best_sign_thr": best_sign_thr,
                    "best_exist_thr": best_exist_thr,
                },
            )

            if early_stop:
                break

        # load the best model
        early_stopping.load_checkpoint(model)
        hyper_parm = early_stopping.load_hyper_param()
        best_sign_thr = hyper_parm["best_sign_thr"]
        best_exist_thr = hyper_parm["best_exist_thr"]
        # E-1: 训练阶段耗时（不含最终测试评估）
        training_time = time.time() - run_start_time

        # evaluate the best model
        logger.info(f"get final performance on dataset {args.dataset_name}...")
        model[0].profiler.enable()  # 开启模型内部的Profiler以记录测试阶段的时间
        inference_start_time = time.time()
        (
            test_losses,
            test_metrics,
            _,
            _,
        ) = evaluate_model_sign_link_3class_prediction(
            model_name=args.model_name,
            model=model,
            neighbor_sampler=full_neighbor_sampler,
            evaluate_idx_data_loader=test_idx_data_loader,
            evaluate_neg_edge_sampler=test_neg_edge_sampler,
            evaluate_data=test_data,
            loss_func=loss_func,
            num_neighbors=args.num_neighbors,
            time_gap=args.time_gap,
            sign_best_thr=best_sign_thr,
            exist_best_thr=best_exist_thr,
        )

        new_node_test_losses, new_node_test_metrics, _, _ = (
            evaluate_model_sign_link_3class_prediction(
                model_name=args.model_name,
                model=model,
                neighbor_sampler=full_neighbor_sampler,
                evaluate_idx_data_loader=new_node_test_idx_data_loader,
                evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                evaluate_data=new_node_test_data,
                loss_func=loss_func,
                num_neighbors=args.num_neighbors,
                time_gap=args.time_gap,
                sign_best_thr=best_sign_thr,
                exist_best_thr=best_exist_thr,
            )
        )
        model[0].profiler.disable()  # 关闭模型内部的Profiler
        # E-1: 推理阶段耗时（test + new node test）
        inference_time = time.time() - inference_start_time
        # store the evaluation metrics at the current run
        (
            val_metric_dict,
            new_node_val_metric_dict,
            test_metric_dict,
            new_node_test_metric_dict,
        ) = ({}, {}, {}, {})

        logger.info(f"test loss: {np.mean(test_losses):.4f}")
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean(
                [test_metric[metric_name] for test_metric in test_metrics]
            )
            logger.info(f"test {metric_name}, {average_test_metric:.4f}")
            test_metric_dict[metric_name] = average_test_metric

        logger.info(f"new node test loss: {np.mean(new_node_test_losses):.4f}")
        for metric_name in new_node_test_metrics[0].keys():
            average_new_node_test_metric = np.mean(
                [
                    new_node_test_metric[metric_name]
                    for new_node_test_metric in new_node_test_metrics
                ]
            )
            logger.info(
                f"new node test {metric_name}, {average_new_node_test_metric:.4f}"
            )
            new_node_test_metric_dict[metric_name] = average_new_node_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f"Run {run + 1} cost {single_run_time:.2f} seconds.")
        # E-1: 峰值显存
        peak_memory_mb = (
            torch.cuda.max_memory_allocated() / 1024 ** 2
            if torch.cuda.is_available()
            else 0.0
        )
        logger.info(f"Run {run + 1} peak memory: {peak_memory_mb:.2f} MB.")

        test_metric_all_runs.append(test_metric_dict)
        new_node_test_metric_all_runs.append(new_node_test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result

        result_json = {
            "test metrics": {
                metric_name: f"{test_metric_dict[metric_name]:.4f}"
                for metric_name in test_metric_dict
            },
            "new node test metrics": {
                metric_name: f"{new_node_test_metric_dict[metric_name]:.4f}"
                for metric_name in new_node_test_metric_dict
            },
            # E-1 效率数据
            "single run time (s)": single_run_time,
            "training time (s)": training_time,
            "inference time (s)": inference_time,
            "peak memory (MB)": peak_memory_mb,
            "parameter count": get_parameter_sizes(model),
            # 设备标记（GPU 统一基座后用于核验表格数据同设备）
            "device": str(args.device),
            # M4：加速状态（口径防混淆；默认启用）
            "accel": _accel.status,
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{TASK_NAME}/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(
            save_result_folder, f"{args.result_save_name}.json"
        )

        with open(save_result_path, "w") as file:
            file.write(result_json)

        save_profiler_folder = f"./saved_results/{TASK_NAME}/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        os.makedirs(save_profiler_folder, exist_ok=True)
        model[0].profiler.save(os.path.join(save_profiler_folder, f"{args.result_save_name}"))
        model[0].profiler.reset()  # 重置之前的记录，确保下一次运行时记录的是新的测试阶段的时间

    # store the average metrics at the log of the last run
    logger.info(f"metrics over {args.num_runs} runs:")

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(
            f"test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}"
        )
        logger.info(
            f"average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} "
            f"± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}"
        )

    for metric_name in new_node_test_metric_all_runs[0].keys():
        logger.info(
            f"new node test {metric_name}, {[new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]}"
        )
        logger.info(
            f"average new node test {metric_name}, {np.mean([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]):.4f} "
            f"± {np.std([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs], ddof=1):.4f}"
        )

    sys.exit()
