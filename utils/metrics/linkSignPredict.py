import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.calibration import label_binarize
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
)
import torch

from utils.metrics import np_sigmoid, np_softmax, safe_roc_auc_score


def get_linksign_prediction_metrics(
    exist_predicts: torch.Tensor,
    exist_labels: torch.Tensor,
    sign_predicts: torch.Tensor,
    sign_labels: torch.Tensor,
    best_exist_thr: float = 0.5,
    best_sign_thr: float = 0.5,
    *,
    reject_support: bool = False,
    is_logits: bool = True
):
    if not reject_support:
        return get_link_sign_3class_prediction_metrics(
            exist_predicts,
            exist_labels,
            sign_predicts,
            sign_labels,
            best_exist_thr,
            best_sign_thr,
            is_logits=is_logits,
        )
    else:
        return get_link_sign_3class_prediction_metrics_support_reject(
            exist_predicts,
            exist_labels,
            sign_predicts,
            sign_labels,
            best_exist_thr,
            is_logits=is_logits,
        )


def get_link_sign_3class_prediction_metrics(
    exist_predicts: torch.Tensor,
    exist_labels: torch.Tensor,
    sign_predicts: torch.Tensor,
    sign_labels: torch.Tensor,
    best_exist_thr: float = 0.5,
    best_sign_thr: float = 0.5,
    *,
    is_logits: bool = True
):
    exist_predicts = exist_predicts.cpu().detach().numpy().ravel()
    exist_labels = exist_labels.cpu().numpy().ravel()
    sign_predicts = sign_predicts.cpu().detach().numpy().ravel()
    sign_labels = sign_labels.cpu().numpy().ravel()

    if is_logits:
        prob_exist = np_sigmoid(exist_predicts)
        prob_sign = np_sigmoid(sign_predicts)
    else:
        prob_exist = exist_predicts
        prob_sign = sign_predicts

    # best_exist_thr = best_thr(prob_exist, exist_labels, True, 0.7)
    # best_sign_thr = best_thr(prob_sign, sign_labels)

    pred_exist = prob_exist >= best_exist_thr
    pred_sign = prob_sign >= best_sign_thr

    n_sign_task = sign_labels.shape[0]
    # 标签生成
    y_true = np.empty(len(exist_labels), dtype=int)
    y_true[:n_sign_task] = sign_labels
    y_true[n_sign_task:] = 2

    # 预测生成
    y_pred = np.empty_like(y_true, dtype=int)

    passed = pred_exist
    passed_real = passed[:n_sign_task]
    n_pass_real = passed_real.sum()

    y_pred[:n_sign_task] = pred_sign.astype(int)
    y_pred[n_sign_task:] = 2
    if n_pass_real > 0:
        # 判断有连边，用符号预测结果
        y_pred[:n_sign_task][passed_real] = pred_sign[passed_real].astype(int)
    # 有连边但是被判断为无连边
    y_pred[:n_sign_task][~passed_real] = 2

    prob_3 = np.zeros((len(y_true), 3))
    prob_3[:n_sign_task, 0] = 1 - prob_sign.squeeze()  # 负边概率，prob_sign 计算
    prob_3[:n_sign_task, 1] = prob_sign.squeeze()  # 正边概率，prob_sign计算
    # null 概率 = 1 - exist 概率
    prob_3[n_sign_task:, 2] = (
        1 - prob_exist[n_sign_task:].squeeze()
    )  # 无连边概率 prob_exist 计算

    labels_bin = label_binarize(y_true, classes=[0, 1, 2])

    # 计算指标
    # 二分类指标
    sign_f1_binary = f1_score(sign_labels, pred_sign, average="binary")
    exist_precision, exist_recall, exist_f1, _ = precision_recall_fscore_support(
        exist_labels, pred_exist, average="binary"
    )
    # exist_f1 = f1_score(sign_labels, pred_sign, average="binary")
    # exist_precision=precision_recall_curve()
    # exist_recall = recall_score(exist_labels, pred_exist)
    # 3分类指标
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)

    acc = accuracy_score(y_true, y_pred)

    average_precision = average_precision_score(
        y_true=labels_bin, y_score=prob_3, average="macro"
    )
    auc = safe_roc_auc_score(y_true=y_true, y_pred=prob_3, average="macro")
    # report = classification_report(y_true, y_pred)
    # print(report)
    return {
        "exist_recall": exist_recall,
        "exist_precision": exist_precision,
        "exist_f1": exist_f1,
        "sign_f1": sign_f1_binary,
        "ap": average_precision,
        "f1_mac": f1_macro,
        "f1_wt": f1_weighted,
        "f1_mic": f1_micro,
        "acc": acc,
        "auc": auc,
    }


def get_link_sign_3class_prediction_metrics_support_reject(
    exist_predicts: torch.Tensor,  # [N, 1]
    exist_labels: torch.Tensor,  # [N]，前一半是1，后一半是0
    sign_predicts: torch.Tensor,  # [N, 3]，前一半是真实符号，后一半是2
    sign_labels: torch.Tensor,  # [N]，前一半是0/1，后一半是2
    best_exist_thr: float = 0.5,
    *,
    is_logits: bool = True
):
    # 转numpy
    exist_logits = exist_predicts.cpu().detach().numpy().ravel()
    exist_labels = exist_labels.cpu().numpy().ravel()
    sign_logits = sign_predicts.cpu().detach().numpy()  # [N, 3]
    sign_labels = sign_labels.cpu().numpy().ravel()

    # 概率
    if is_logits:
        prob_exist = np_sigmoid(exist_logits)  # [N]
        prob_sign = np_softmax(sign_logits, axis=-1)  # [N, 3]
    else:
        prob_exist = exist_logits
        prob_sign = sign_logits

    # 预测
    pred_exist = prob_exist > best_exist_thr  # [N]
    pred_sign = prob_sign.argmax(axis=-1)  # [N]，0/1/2

    # === 级联预测 ===
    # 默认不存在（2）
    y_pred = np.full_like(exist_labels, 2)

    # 第一步通过的，用第二步预测
    passed = pred_exist
    y_pred[passed] = pred_sign[passed]

    # 但第二步预测为2（拒绝）的，也改为不存在（2）→ 已经是2，不用改

    y_true = sign_labels  # 直接用，已经是0/1/2

    # === 指标 ===
    exist_f1 = f1_score(exist_labels, pred_exist, average="binary")

    # 符号F1：只在真实存在（标签0/1）且第一步通过的样本上算
    real_mask = sign_labels < 2  # 0或1
    passed_real = pred_exist & real_mask
    if passed_real.sum() > 0:
        sign_f1 = f1_score(
            sign_labels[passed_real],
            y_pred[passed_real],
            labels=[0, 1],
            average="macro",
        )
    else:
        sign_f1 = 0.0

    # 整体3分类F1
    f1_macro = f1_score(sign_labels, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    # 拒绝准确率（null样本被分到2的比例）
    null_mask = sign_labels == 2
    reject_acc = (y_pred[null_mask] == 2).mean() if null_mask.any() else 0.0

    return {
        "exist_f1": exist_f1,
        "sign_f1": sign_f1,
        "f1_mac": f1_macro,
        "f1_wt": f1_weighted,
        "f1_mic": f1_micro,
        "reject_acc": reject_acc,
        "acc": acc,
    }
