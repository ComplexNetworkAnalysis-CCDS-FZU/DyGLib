"""
符号预测指标。实际复用通用二分类指标，额外补上 f1_weighted。
"""
import numpy as np
from sklearn.metrics import f1_score
import torch

from utils.metrics.binary_classification import get_binary_classification_metrics


def get_sign_prediction_metrics(
    predicts: torch.Tensor,
    labels: torch.Tensor,
    *,
    thr: float = 0.5,
    is_logits: bool = True,
) -> dict:
    metrics = get_binary_classification_metrics(
        predicts=predicts, labels=labels, thr=thr, is_logits=is_logits,
    )
    # 保持向后兼容: 补充 f1_weighted
    if is_logits:
        predicts = predicts.sigmoid()
    y_score = (predicts.cpu().detach().numpy() >= thr).astype(int)
    labels_np = labels.cpu().numpy()
    if len(np.unique(labels_np)) >= 2:
        metrics["f1_weighted"] = f1_score(
            y_true=labels_np, y_pred=y_score, average="weighted", zero_division=0,
        )
    else:
        metrics["f1_weighted"] = 0.0
    return metrics
