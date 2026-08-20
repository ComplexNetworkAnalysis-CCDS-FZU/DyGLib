"""通用二分类评价指标。符号预测、链路预测均可复用。"""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from sklearn.calibration import label_binarize

from utils.metrics import safe_roc_auc_score


def get_binary_classification_metrics(
    predicts: torch.Tensor,
    labels: torch.Tensor,
    *,
    thr: float = 0.5,
    is_logits: bool = True,
) -> dict:
    """
    通用二分类指标: AUC, AP, Acc, F1_binary, F1_macro。
    :param predicts: Tensor, shape (N,)
    :param labels:   Tensor, shape (N,)
    :param thr:      分类阈值 (若外部已做自适应阈值则传入)
    :param is_logits: predicts 是否为 logits (True 则先 sigmoid)
    """
    if is_logits:
        predicts = predicts.sigmoid()

    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    y_score = (predicts >= thr).astype(int)
    acc = accuracy_score(labels, y_pred=y_score)

    if len(np.unique(labels)) < 2:
        return {"ap": 0.0, "f1_macro": 0.0, "f1_binary": 0.0, "acc": acc, "auc": 0.5}

    f1_macro = f1_score(y_true=labels, y_pred=y_score, zero_division=0, average="macro")
    f1_binary = f1_score(y_true=labels, y_pred=y_score, zero_division=0)
    labels_bin = label_binarize(labels, classes=[0, 1])
    ap = average_precision_score(y_true=labels_bin, y_score=predicts, average="macro")
    auc = safe_roc_auc_score(y_true=labels_bin, y_pred=predicts, average="weight")

    return {
        "ap": ap,
        "f1_macro": f1_macro,
        "f1_binary": f1_binary,
        "acc": acc,
        "auc": auc,
    }
