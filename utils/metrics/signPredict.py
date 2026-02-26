import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.calibration import label_binarize
from sklearn.metrics import average_precision_score, f1_score
import torch

from utils.metrics import safe_roc_auc_score


def get_sign_prediction_metrics(
    predicts: torch.Tensor,
    labels: torch.Tensor,
    *,
    thr: float = 0.5,
    is_logits: bool = True
):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    if is_logits:
        predicts = predicts.sigmoid()
    else:
        predicts = predicts

    predicts = predicts.cpu().detach().numpy()
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
