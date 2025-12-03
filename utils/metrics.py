from typing import Literal, Optional
import numpy as np
from sklearn.calibration import label_binarize
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    accuracy_score,
)


def save_roc_auc_score(
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

    #二分类短路
    if n_classes == 2 and y_pred.ndim == 1:
        return roc_auc_score(y_true=y_true,y_score=y_pred)


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


def best_thr(predict:np.ndarray,labels:np.ndarray):
    precision, recall, thr = precision_recall_curve(labels, predict)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx  = np.argmax(f1_scores) 
    best_thr  = thr[best_idx]  

    return best_thr

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
    predicts: torch.Tensor, labels: torch.Tensor
):
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

    f1_macro = f1_score(
        y_true=labels,
        y_pred=y_score,
        zero_division=0,
        average="macro",
    )

    acc = accuracy_score(labels, y_pred=y_score)

    labels_bin = label_binarize(labels, classes=[0, 1, 2])
    average_precision = average_precision_score(
        y_true=labels_bin, y_score=predicts, average="macro"
    )
    auc = save_roc_auc_score(y_true=labels_bin, y_score=predicts, average="weight")

    return {"AP": average_precision, "F1": f1_macro, "acc": acc, "auc": auc}

def get_sign_prediction_metrics(
    predicts: torch.Tensor, labels: torch.Tensor,*,thr:float = 0.5
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


    y_score = (predicts >=thr).astype(int) 

    f1_macro = f1_score(
        y_true=labels,
        y_pred=y_score,
        zero_division=0,
        average="macro",
    )

    acc = accuracy_score(labels, y_pred=y_score)

    labels_bin = label_binarize(labels, classes=[0, 1])
    average_precision = average_precision_score(
        y_true=labels_bin, y_score=predicts, average="macro"
    )
    auc = save_roc_auc_score(y_true=labels_bin, y_pred=predicts, average="weight")

    return {"AP": average_precision, "F1": f1_macro, "acc": acc, "auc": auc}


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
