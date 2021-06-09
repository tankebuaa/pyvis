import torch
import torch.nn.functional as F
from model.loss.iou_loss import linear_iou


def log_softmax(cls):
    cls = cls.permute(0, 2, 3, 1).contiguous()
    cls = F.log_softmax(cls, dim=3)
    return cls


def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)


def pos_cross_entropy_loss(pred, label):
    pred = log_softmax(pred)
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    return loss_pos


def select_cross_entropy_loss(pred, label):
    pred = log_softmax(pred)
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def select_iou_loss(pred_loc, label_loc, weights):
    iou_loss = linear_iou(pred_loc, label_loc, weights)
    return iou_loss


def weight_l2_loss(pred_iou, label_iou, loss_weight):
    b, n = pred_iou.size()
    diff = (pred_iou - label_iou).pow(2)
    diff = diff.sum(dim=1)
    loss = diff * loss_weight.squeeze()
    return loss.sum().div(b * n)
