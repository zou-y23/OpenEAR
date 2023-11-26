#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetCrossEntropy(nn.Module):
    """
    Cross entropy loss with soft target.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(SoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        loss = torch.sum(-y * F.log_softmax(x, dim=-1), dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError


_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": SoftTargetCrossEntropy,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]


class EvidenceLoss(nn.Module):
    def __init__(self, num_cls):
        super(EvidenceLoss, self).__init__()
        self.num_cls = num_cls
        self.loss_type = 'log'
        self.evidence = 'exp'

    def forward(self, logit, target):
        """ logit, shape=(N, K+1)
            target, shape=(N, 1)
        """
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            logit = logit.view(-1, logit.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
        #target = target.view(-1)  # [N,d1,d2,...]->[N*d1*d2*...,]

        out_dict = dict()

        # one-hot embedding for the target
        # y = torch.eye(self.num_cls).to(logit.device, non_blocking=True)
        # y = y[target]  # (N, K+1)
        y = target
        
        # get loss func
        loss, func = self.get_loss_func()
        
        # compute losses
        pred_alpha = self.evidence_func(logit) + 1  # (alpha = e + 1)
        loss_out = loss(y, pred_alpha, func=func)
        out_dict.update(loss_out)

        # accumulate total loss
        total_loss = 0
        for k, v in loss_out.items():
            if 'loss' in k:
                total_loss += v
        out_dict.update({'total_loss': total_loss})
        return total_loss


    def get_loss_func(self):
        if self.loss_type == 'log':
            return self.edl_loss, torch.log
        elif self.loss_type == 'digamma':
            return self.edl_loss, torch.digamma
        else:
            raise NotImplementedError


    def evidence_func(self, logit):
        if self.evidence == 'relu':
            return F.relu(logit)

        if self.evidence == 'exp':
            return torch.exp(torch.clamp(logit, -10, 10))

        if self.evidence == 'softplus':
            return F.softplus(logit)
        

    def edl_loss(self, y, alpha, func=torch.log):
        losses = {}
        S = torch.sum(alpha, dim=1, keepdim=True)  # (B, 1)
        #print(y.shape, S.shape, alpha.shape)
        cls_loss = torch.sum(y * (func(S) - func(alpha)), dim=1)
        cls_loss = torch.sum(cls_loss)
        losses.update({'cls_loss': cls_loss})
        return losses
