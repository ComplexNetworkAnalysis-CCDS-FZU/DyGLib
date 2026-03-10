from typing import Optional
from sklearn.metrics import precision_recall_curve
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
import time
import argparse
import os
import json

from models.DyGFormer import DyGFormer
from models.EdgeBank import edge_bank_link_prediction
from models.SignDyGFormer import SignDyGFormer
from models.modules import cascade_loss, sign_link3class_label

from utils.metrics import best_thr
from utils.metrics.linkSignPredict import (
    cascade_best_thr,
    get_linksign_prediction_metrics,
)
from utils.metrics.signPredict import get_sign_prediction_metrics
from utils.utils import set_random_seed
from utils.utils import NegativeEdgeSampler, NeighborSampler
from utils.DataLoader import Data


def evaluate_model_sign_link_3class_prediction(
    model_name: str,
    model: nn.Module,
    neighbor_sampler: NeighborSampler,
    evaluate_idx_data_loader: DataLoader,
    evaluate_neg_edge_sampler: NegativeEdgeSampler,
    evaluate_data: Data,
    loss_func: nn.Module,
    num_neighbors: int = 20,
    time_gap: int = 2000,
    exist_best_thr: Optional[float] = None,
    sign_best_thr: Optional[float] = None,
    *,
    reject_support: bool = False,
):
    """
    evaluate models on the link sign prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    if model_name in [
        DyGFormer.NAME,
        SignDyGFormer.NAME,
    ]:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        all_predict, all_label = [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(
            evaluate_idx_data_loader_tqdm
        ):
            evaluate_data_indices = evaluate_data_indices.numpy()
            (
                batch_src_node_ids,
                batch_dst_node_ids,
                batch_node_interact_times,
                batch_edge_ids,
                batch_node_interact_sign,
            ) = (
                evaluate_data.src_node_ids[evaluate_data_indices],
                evaluate_data.dst_node_ids[evaluate_data_indices],
                evaluate_data.node_interact_times[evaluate_data_indices],
                evaluate_data.edge_ids[evaluate_data_indices],
                evaluate_data.node_interact_sign[evaluate_data_indices],
            )

            mask = batch_node_interact_sign.squeeze() != 0

            batch_src_node_ids = batch_src_node_ids[mask]
            batch_dst_node_ids = batch_dst_node_ids[mask]
            batch_node_interact_times = batch_node_interact_times[mask]
            batch_edge_ids = batch_edge_ids[mask]
            batch_node_interact_sign = batch_node_interact_sign[mask]

            if evaluate_neg_edge_sampler.negative_sample_strategy != "random":
                batch_neg_src_node_ids, batch_neg_dst_node_ids = (
                    evaluate_neg_edge_sampler.sample(
                        size=len(batch_src_node_ids),
                        batch_src_node_ids=batch_src_node_ids,
                        batch_dst_node_ids=batch_dst_node_ids,
                        current_batch_start_time=batch_node_interact_times[0],
                        current_batch_end_time=batch_node_interact_times[-1],
                    )
                )
            else:
                _, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(
                    size=len(batch_src_node_ids)
                )
                batch_neg_src_node_ids = batch_src_node_ids

            if model_name in [DyGFormer.NAME]:
                # get temporal embedding of source and destination nodes
                batch_src_node_embeddings, batch_dst_node_embeddings = model[
                    0
                ].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=batch_src_node_ids,
                    dst_node_ids=batch_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                )

                # get temporal embedding of negative source and negative destination nodes
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = model[
                    0
                ].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=batch_neg_src_node_ids,
                    dst_node_ids=batch_neg_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                )
            elif model_name in [SignDyGFormer.NAME]:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = model[
                    0
                ].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=batch_src_node_ids,
                    dst_node_ids=batch_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                    node_interact_sign=batch_node_interact_sign,
                )

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = model[
                    0
                ].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=batch_neg_src_node_ids,
                    dst_node_ids=batch_neg_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                    node_interact_sign=np.zeros_like(batch_node_interact_sign),
                )
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
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
                edge_sign=batch_node_interact_sign,
                reject_support=reject_support,
            )

            loss = cascade_loss(
                exist_predict,
                sign_predict,
                exist_prob,
                y_exist,
                y_sign,
                reject_support=reject_support,
            )

            evaluate_losses.append(loss.item())

            all_predict.append(
                (torch.sigmoid(exist_predict), torch.sigmoid(sign_predict))
            )
            all_label.append((y_exist, y_sign))

            evaluate_idx_data_loader_tqdm.set_description(
                f"evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}"
            )

        if exist_best_thr is None and sign_best_thr is None:
            exist_prob = (
                torch.cat([v[0] for v in all_predict]).squeeze(-1).cpu().numpy()
            )  # [N_val]
            exist_label = torch.cat([v[0] for v in all_label]).squeeze(-1).cpu().numpy()
            sign_prob = (
                torch.cat([v[1] for v in all_predict]).squeeze(-1).cpu().numpy()
            )  # [N_real]
            sign_label = torch.cat([v[1] for v in all_label]).squeeze(-1).cpu().numpy()

            exist_best_thr, sign_best_thr = cascade_best_thr(
                exist_predicts=exist_prob,
                exist_labels=exist_label,
                sign_predicts=sign_prob,
                sign_labels=sign_label,
                is_logits=False,
            )

        if exist_best_thr is None:
            exist_prob = (
                torch.cat([v[0] for v in all_predict]).squeeze(-1).cpu().numpy()
            )  # [N_val]
            exist_label = torch.cat([v[0] for v in all_label]).squeeze(-1).cpu().numpy()
            exist_best_thr = float(best_thr(exist_prob, exist_label, grid=True))

        if sign_best_thr is None:
            sign_prob = (
                torch.cat([v[1] for v in all_predict]).squeeze(-1).cpu().numpy()
            )  # [N_real]
            sign_label = torch.cat([v[1] for v in all_label]).squeeze(-1).cpu().numpy()

            sign_best_thr = float(best_thr(sign_prob, sign_label, min_recall=0.95))

        for (exist_predict, sign_predict), (exist_label, sign_label) in zip(
            all_predict, all_label
        ):

            evaluate_metrics.append(
                get_linksign_prediction_metrics(
                    sign_predicts=sign_predict,
                    sign_labels=sign_label,
                    exist_predicts=exist_predict,
                    exist_labels=exist_label,
                    best_exist_thr=exist_best_thr,
                    best_sign_thr=sign_best_thr,
                    reject_support=reject_support,
                    is_logits=False,
                )
            )

    return evaluate_losses, evaluate_metrics, exist_best_thr, sign_best_thr


def evaluate_model_sign_prediction(
    model_name: str,
    model: nn.Module,
    neighbor_sampler: NeighborSampler,
    evaluate_idx_data_loader: DataLoader,
    evaluate_data: Data,
    loss_func: nn.Module,
    num_neighbors: int = 20,
    time_gap: int = 2000,
    thr: Optional[float] = None,
):
    """
    evaluate models on the link sign prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)

    if model_name in [
        DyGFormer.NAME,
        SignDyGFormer.NAME,
    ]:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        all_predict, all_label = [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(
            evaluate_idx_data_loader_tqdm
        ):
            evaluate_data_indices = evaluate_data_indices.numpy()
            (
                batch_src_node_ids,
                batch_dst_node_ids,
                batch_node_interact_times,
                batch_edge_ids,
                batch_node_interact_sign,
            ) = (
                evaluate_data.src_node_ids[evaluate_data_indices],
                evaluate_data.dst_node_ids[evaluate_data_indices],
                evaluate_data.node_interact_times[evaluate_data_indices],
                evaluate_data.edge_ids[evaluate_data_indices],
                evaluate_data.node_interact_sign[evaluate_data_indices],
            )

            mask = batch_node_interact_sign.squeeze() != 0

            batch_src_node_ids = batch_src_node_ids[mask]
            batch_dst_node_ids = batch_dst_node_ids[mask]
            batch_node_interact_times = batch_node_interact_times[mask]
            batch_edge_ids = batch_edge_ids[mask]
            batch_node_interact_sign = batch_node_interact_sign[mask]

            if model_name in [DyGFormer.NAME]:
                # get temporal embedding of source and destination nodes
                batch_src_node_embeddings, batch_dst_node_embeddings = model[
                    0
                ].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=batch_src_node_ids,
                    dst_node_ids=batch_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                )

            elif model_name in [SignDyGFormer.NAME]:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = model[
                    0
                ].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=batch_src_node_ids,
                    dst_node_ids=batch_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                    node_interact_sign=batch_node_interact_sign,
                )

            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            # get positive and negative probabilities, shape (batch_size, )
            positive_probabilities = model[1](
                input_1=batch_src_node_embeddings,
                input_2=batch_dst_node_embeddings,
            ).squeeze(-1)

            # 过滤掉中立交互的情况

            positive_probabilities_filter = positive_probabilities

            negative_probabilities = positive_probabilities_filter[
                batch_node_interact_sign == -1
            ]

            positive_probabilities = positive_probabilities_filter[
                batch_node_interact_sign == 1
            ]

            predicts = torch.cat(
                [
                    positive_probabilities,
                    negative_probabilities,
                ],
                dim=0,
            )
            labels = torch.cat(
                [
                    torch.ones(
                        positive_probabilities.size(0),
                        device=positive_probabilities.device,
                        dtype=torch.float,
                    ),
                    torch.zeros(
                        negative_probabilities.size(0),
                        device=negative_probabilities.device,
                        dtype=torch.float,
                    ),
                ],
            )

            all_predict.append(predicts.sigmoid().numpy())
            all_label.append(labels.numpy())

            loss = loss_func(predicts, labels)

            evaluate_losses.append(loss.item())

            evaluate_idx_data_loader_tqdm.set_description(
                f"evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}"
            )

        # 计算best thr
        if thr is None:
            val_pred = np.concatenate(all_predict)
            val_true = np.concatenate(all_label)

            thr = best_thr(val_pred, val_true)
            print("prob 分布", np.percentile(val_pred, [0, 1, 10, 50, 90, 99, 100]))
            precision, recall, thrs = precision_recall_curve(val_true, val_pred)
            print(f"precision: {precision}, recall: {recall}")
            f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
            best_idx = np.argmax(f1_scores)
            print(f"max F1 = {f1_scores[best_idx]:.3f} @ thr = {thrs[best_idx]:.3f}")
            print(
                f"your thr= {thr:.3f} → F1= {f1_scores[np.searchsorted(thrs, thr)]:.3f}"
            )

            print("pos prob", val_pred[val_true == 1].mean())
            print("neg prob", val_pred[val_true == 0].mean())

            y_pred = (val_pred >= thr).astype(int)
            print("测试集 pred 正例数", y_pred.sum())
            print("测试集 true 正例数", val_true.sum())
            print("pred 正例 / true 正例 =", y_pred.sum() / max(val_true.sum(), 1))

        print(f"best thr: {thr}")

        for val_pred, val_true in zip(all_predict, all_label):
            evaluate_metrics.append(
                get_sign_prediction_metrics(
                    predicts=torch.tensor(val_pred),
                    labels=torch.tensor(val_true),
                    thr=thr,
                    is_logits=False,
                )
            )

    return evaluate_losses, evaluate_metrics, thr
