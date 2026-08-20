"""有向链路预测指标。复用通用二分类指标。"""
from utils.metrics.binary_classification import get_binary_classification_metrics


def get_link_prediction_metrics(predicts, labels, *, thr=0.5, is_logits=True):
    return get_binary_classification_metrics(
        predicts=predicts, labels=labels, thr=thr, is_logits=is_logits,
    )
