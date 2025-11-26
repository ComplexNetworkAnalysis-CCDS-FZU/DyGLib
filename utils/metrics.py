from sklearn.calibration import label_binarize
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


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

    return {'average_precision': average_precision, 'roc_auc': roc_auc}

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
    f1 = f1_score(y_true=labels, y_pred=y_score,zero_division=0,average='macro',)

    labels =label_binarize(labels, classes=[0, 1, 2, 3])
    average_precision = average_precision_score(y_true=labels, y_score=predicts,average='macro')

    return {'average_precision': average_precision, 'f1': f1}

def get_link_sign_3class_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
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
    f1 = f1_score(y_true=labels, y_pred=y_score,zero_division=0,average='macro',)

    labels =label_binarize(labels, classes=[0, 1, 2])
    average_precision = average_precision_score(y_true=labels, y_score=predicts,average='macro')

    return {'average_precision': average_precision, 'f1': f1}

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

    return {'roc_auc': roc_auc}
