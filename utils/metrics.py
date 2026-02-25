from typing import Literal, Optional, Tuple
import numpy as np
from sklearn.calibration import label_binarize
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
    accuracy_score,
)

def np_softmax(x, axis=-1):
    """数值稳定的 softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def safe_roc_auc_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    average: Literal["weighted", "micro", "macro"] = "weighted",
    n_classes: Optional[int] = None,
):
    if n_classes is None:
        if y_pred.ndim == 1:
            n_classes = 2
        else:
            n_classes = y_pred.shape[1]

    # 二分类短路
    if n_classes == 2 and y_pred.ndim == 1:
        return roc_auc_score(y_true=y_true, y_score=y_pred)

    # 判断类型一致
    assert (
        n_classes == y_pred.shape[1]
    ), f"预测类数量[{y_pred.shape[1]}]与给定类数量[{n_classes}]不一致"

    auc = []
    weight = []

    for c in range(n_classes):
        # 分为 c 类 和非c 类
        num_y_true = (y_true == c).astype(np.int32)
        if len(np.unique(num_y_true)) < 2:
            # 只有单类，无法分类，跳过
            continue
        # 计算单类的AUC
        auc.append(roc_auc_score(num_y_true, y_pred[:, c]))

        # 权重统计
        if average == "weighted":
            weight.append((y_true == c).sum())
        else:
            weight.append(1.0)

    if len(auc) == 0:
        # 全都是单类，跳了
        return np.nan
    elif average in ["weight", "macro"]:
        return np.average(auc, weights=weight)
    else:
        return roc_auc_score(y_true, y_pred, average="micro", n_classes=n_classes)


def best_thr(
    predict: np.ndarray,
    labels: np.ndarray,
    *,
    best_recall=False,
    min_recall: Optional[float] = None,
    print_f1: bool = True,
):
    precision, recall, thr = precision_recall_curve(labels, predict)
    if best_recall:
        if min_recall is not None:
            valid = recall >= min_recall
            best_idx = np.argmax(precision[valid]) if valid.any() else np.argmax(recall)
        else:
            best_idx = np.argmax(recall)
    else:
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        if print_f1:
            print(f"最优阈值: {thr[best_idx]:.3f}")
            print(f"Precision: {precision[best_idx]:.3f}")
            print(f"Recall: {recall[best_idx]:.3f}")
            print(f"F1: {f1_scores[best_idx]:.3f}")
    best_thr = thr[best_idx]

    return best_thr


def joint_pred(
    exist_predicts: np.ndarray,
    sign_predicts: np.ndarray,
    exist_thr: float,
    sign_thr: float,
):
    has = exist_predicts >= exist_thr  # bool
    sign = sign_predicts >= sign_thr  # bool
    return np.where(has, sign.astype(int) + 1, 0)


def best_thr_fast(
    exist_predicts: np.ndarray,
    sign_predicts: np.ndarray,
    labels: np.ndarray,
    coarse=20,
    fine=50,
    radius=0.1,
    average: Literal["macro", "micro", "weight", "binary"] = "macro",
):
    # 1. 粗搜
    c1 = np.linspace(0.01, 0.99, coarse)
    c2 = np.linspace(0.01, 0.99, coarse)
    f1_coarse = np.zeros((coarse, coarse))
    for i, t1 in enumerate(c1):
        for j, t2 in enumerate(c2):
            pred = joint_pred(exist_predicts, sign_predicts, t1, t2)  # 一行函数见下
            f1_coarse[i, j] = f1_score(labels, pred, average=average)
    idx = np.unravel_index(f1_coarse.argmax(), f1_coarse.shape)
    t1_c, t2_c = c1[idx[0]], c2[idx[1]]

    # 2. 精搜
    f1_fine = np.zeros((fine, fine))
    fine1 = np.linspace(max(0.01, t1_c - radius), min(0.99, t1_c + radius), fine)
    fine2 = np.linspace(max(0.01, t2_c - radius), min(0.99, t2_c + radius), fine)
    for i, t1 in enumerate(fine1):
        for j, t2 in enumerate(fine2):
            pred = joint_pred(exist_predicts, sign_predicts, t1, t2)
            f1_fine[i, j] = f1_score(labels, pred, average=average)
    idx = np.unravel_index(f1_fine.argmax(), f1_fine.shape)
    return fine1[idx[0]], fine2[idx[1]]


def best_cascade_thr(
    exist_predicts: np.ndarray,
    exist_labels: np.ndarray,
    sign_predicts: np.ndarray,
    sign_labels: np.ndarray,
) -> Tuple[float, float]:
    # ---- 拼三分类标签 ----
    y_true = np.empty_like(exist_labels)
    y_true[exist_labels == 0] = 0
    mask = exist_labels == 1
    y_true[mask] = sign_labels[mask] + 1  # 0→1  1→2

    exist_thr, sign_thr = best_thr_fast(exist_predicts, sign_predicts, y_true)

    return float(exist_thr), float(sign_thr)


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {"average_precision": average_precision, "roc_auc": roc_auc}


def get_link_sign_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    y_score = predicts.argmax(1)
    f1 = f1_score(
        y_true=labels,
        y_pred=y_score,
        zero_division=0,
        average="macro",
    )

    labels = label_binarize(labels, classes=[0, 1, 2, 3])
    average_precision = average_precision_score(
        y_true=labels, y_score=predicts, average="macro"
    )

    return {"average_precision": average_precision, "f1": f1}


def get_link_sign_3class_prediction_metrics(
    exist_predicts: torch.Tensor,
    exist_labels: torch.Tensor,
    sign_predicts: torch.Tensor,
    sign_labels: torch.Tensor,
    best_exist_thr: float = 0.5,
    best_sign_thr: float = 0.5,
):
    exist_predicts = exist_predicts.cpu().detach().numpy().ravel()
    exist_labels = exist_labels.cpu().numpy().ravel()
    sign_predicts = sign_predicts.cpu().detach().numpy().ravel()
    sign_labels = sign_labels.cpu().numpy().ravel()

    prob_exist = np_sigmoid(exist_predicts)
    prob_sign = np_sigmoid(sign_predicts)

    # best_exist_thr = best_thr(prob_exist, exist_labels, True, 0.7)
    # best_sign_thr = best_thr(prob_sign, sign_labels)

    pred_exist = prob_exist > best_exist_thr
    pred_sign = prob_sign > best_sign_thr

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
):
    # 转numpy
    exist_logits = exist_predicts.cpu().detach().numpy().ravel()
    exist_labels = exist_labels.cpu().numpy().ravel()
    sign_logits = sign_predicts.cpu().detach().numpy()  # [N, 3]
    sign_labels = sign_labels.cpu().numpy().ravel()

    # 概率
    prob_exist = np_sigmoid(exist_logits)  # [N]
    prob_sign = np_softmax(sign_logits, axis=-1)  # [N, 3]

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


def get_sign_prediction_metrics(
    predicts: torch.Tensor, labels: torch.Tensor, *, thr: float = 0.5
):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.sigmoid().cpu().detach().numpy()
    labels = labels.cpu().numpy()

    y_score = (predicts >= thr).astype(int)
    acc = accuracy_score(labels, y_pred=y_score)

    if len(np.unique(labels)) < 2:
        return {
            "ap": 0.0,
            "f1_macro": 0.0,
            "f1_binary": 0.0,
            "f1_weighted": 0.0,
            "acc": acc,
            "auc": 0.5,
        }

    f1_macro = f1_score(
        y_true=labels,
        y_pred=y_score,
        zero_division=0,
        average="macro",
    )

    f1_binary = f1_score(y_true=labels, y_pred=y_score, zero_division=0)

    f1_wt = f1_score(y_true=labels, y_pred=y_score, average="weighted", zero_division=0)
    labels_bin = label_binarize(labels, classes=[0, 1])
    average_precision = average_precision_score(
        y_true=labels_bin, y_score=predicts, average="macro"
    )
    auc = safe_roc_auc_score(y_true=labels_bin, y_pred=predicts, average="weight")

    return {
        "ap": average_precision,
        "f1_macro": f1_macro,
        "f1_binary": f1_binary,
        "f1_weighted": f1_wt,
        "acc": acc,
        "auc": auc,
    }


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {"roc_auc": roc_auc}
