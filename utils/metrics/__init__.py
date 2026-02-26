from typing import Literal, Optional

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from torch import Tuple


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
    grid: bool = False,
    grid_step: float = 0.001,
):
    if grid:
        thresholds = np.arange(grid_step, 1.0, grid_step)
        best_f1, best_thresh = 0, 0.5

        for t in thresholds:
            pred = (predict >= t).astype(int)
            f1 = f1_score(labels, pred, average="macro", zero_division=0)

            if f1 > best_f1:
                best_f1, best_thresh = f1, t

        if print_f1:
            print("best F1", best_f1)
            print("best thr", best_thresh)
        return best_thresh

    precision, recall, thr = precision_recall_curve(labels, predict)
    if best_recall:
        if min_recall is not None:
            valid = recall >= min_recall
            best_idx = np.argmax(precision[valid]) if valid.any() else np.argmax(recall)
        else:
            best_idx = np.argmax(recall)
    else:
        f1_scores = 2 * precision[1:] * recall[1:] / (precision[1:] + recall[1:] + 1e-8)
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
