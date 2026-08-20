"""
链路预测损失函数。
支持无向/有向两种模式，自动按样本拼接构造标签。
"""
from enum import Enum
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinkLossType(Enum):
    BCE = "bce"
    CROSS_ENTROPY = "cross_entropy"
    FOCAL = "focal"


class EdgeDirectionMode(Enum):
    UNDIRECTED = "undirected"
    DIRECTED = "directed"


class LinkLoss(nn.Module):
    """
    链路预测损失。根据方向模式自动构造标签。

    无向模式:
        pos_logits + neg_logits → [1]*N_pos + [0]*N_neg
        neg 来自随机替换 dst

    有向模式:
        pos_logits + rev_logits + neg_logits → [1]*N_pos + [0]*(N_rev+N_neg)
        rev 是批次中反向边预测（若存在），neg 来自随机采样
    """

    def __init__(
        self,
        loss_type: LinkLossType = LinkLossType.BCE,
        direction: EdgeDirectionMode = EdgeDirectionMode.UNDIRECTED,
        *,
        pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.direction = direction

        if loss_type == LinkLossType.BCE:
            kwargs = {}
            if pos_weight is not None:
                kwargs["pos_weight"] = torch.tensor([pos_weight])
            self._fn = nn.BCEWithLogitsLoss(**kwargs)
        elif loss_type == LinkLossType.CROSS_ENTROPY:
            self._fn = nn.CrossEntropyLoss()
        elif loss_type == LinkLossType.FOCAL:
            self._fn = _FocalLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(
        self,
        *,
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
        rev_logits: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ):
        """
        :param return_logits: 若为 True，返回 (loss, logits, labels) 三元组，方便外部算 metrics
        """
        if self.direction == EdgeDirectionMode.UNDIRECTED and rev_logits is not None:
            raise ValueError(
                f"LinkLoss(direction={self.direction}): rev_logits must be None."
            )

        ones_pos = torch.ones(pos_logits.size(0), device=pos_logits.device, dtype=torch.float)
        zeros_neg = torch.zeros(neg_logits.size(0), device=neg_logits.device, dtype=torch.float)

        if self.direction == EdgeDirectionMode.DIRECTED and rev_logits is not None and rev_logits.size(0) > 0:
            zeros_rev = torch.zeros(rev_logits.size(0), device=rev_logits.device, dtype=torch.float)
            logits = torch.cat([pos_logits, rev_logits, neg_logits], dim=0)
            labels = torch.cat([ones_pos, zeros_rev, zeros_neg], dim=0)
        else:
            logits = torch.cat([pos_logits, neg_logits], dim=0)
            labels = torch.cat([ones_pos, zeros_neg], dim=0)

        loss = self._fn(logits.squeeze(-1), labels)

        if return_logits:
            return loss, logits, labels
        return loss


class _FocalLoss(nn.Module):
    """Focal Loss for binary classification."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t).pow(self.gamma)
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight
        loss = focal_weight * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
