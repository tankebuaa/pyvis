import torch
from torch import nn
import math


def ciou(bboxes1, bboxes2):
    w1 = bboxes1[:, 0] + bboxes1[:, 2]
    h1 = bboxes1[:, 1] + bboxes1[:, 3]
    w2 = bboxes2[:, 0] + bboxes2[:, 2]
    h2 = bboxes2[:, 1] + bboxes2[:, 3]
    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = 0.5 * (bboxes1[:, 2] - bboxes1[:, 0])
    center_y1 = 0.5 * (bboxes1[:, 3] - bboxes1[:, 1])
    center_x2 = 0.5 * (bboxes2[:, 2] - bboxes2[:, 0])
    center_y2 = 0.5 * (bboxes2[:, 3] - bboxes2[:, 1])

    inter_l = torch.max(center_x1 - w1 / 2, center_x2 - w2 / 2)
    inter_r = torch.min(center_x1 + w1 / 2, center_x2 + w2 / 2)
    inter_t = torch.max(center_y1 - h1 / 2, center_y2 - h2 / 2)
    inter_b = torch.min(center_y1 + h1 / 2, center_y2 + h2 / 2)
    inter_area = torch.clamp((inter_r - inter_l), min=0) * torch.clamp((inter_b - inter_t), min=0)

    c_l = torch.min(center_x1 - w1 / 2, center_x2 - w2 / 2)
    c_r = torch.max(center_x1 + w1 / 2, center_x2 + w2 / 2)
    c_t = torch.min(center_y1 - h1 / 2, center_y2 - h2 / 2)
    c_b = torch.max(center_y1 + h1 / 2, center_y2 + h2 / 2)

    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    c_diag = torch.clamp((c_r - c_l), min=0) ** 2 + torch.clamp((c_b - c_t), min=0) ** 2

    union = area1 + area2 - inter_area
    u = (inter_diag) / c_diag
    iou = inter_area / union
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
    with torch.no_grad():
        S = (iou > 0.5).float()
        alpha = S * v / (1 - iou + v)
    cious = iou - u - alpha * v
    cious = torch.clamp(cious, min=-1.0, max=1.0)
    losses = 1 - cious
    return losses.mean()


def diou(bboxes1, bboxes2):
    w1 = bboxes1[:, 0] + bboxes1[:, 2]
    h1 = bboxes1[:, 1] + bboxes1[:, 3]
    w2 = bboxes2[:, 0] + bboxes2[:, 2]
    h2 = bboxes2[:, 1] + bboxes2[:, 3]
    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = 0.5 * (bboxes1[:, 2] - bboxes1[:, 0])
    center_y1 = 0.5 * (bboxes1[:, 3] - bboxes1[:, 1])
    center_x2 = 0.5 * (bboxes2[:, 2] - bboxes2[:, 0])
    center_y2 = 0.5 * (bboxes2[:, 3] - bboxes2[:, 1])

    inter_l = torch.max(center_x1 - w1 / 2, center_x2 - w2 / 2)
    inter_r = torch.min(center_x1 + w1 / 2, center_x2 + w2 / 2)
    inter_t = torch.max(center_y1 - h1 / 2, center_y2 - h2 / 2)
    inter_b = torch.min(center_y1 + h1 / 2, center_y2 + h2 / 2)
    inter_area = torch.clamp((inter_r - inter_l), min=0) * torch.clamp((inter_b - inter_t), min=0)

    c_l = torch.min(center_x1 - w1 / 2, center_x2 - w2 / 2)
    c_r = torch.max(center_x1 + w1 / 2, center_x2 + w2 / 2)
    c_t = torch.min(center_y1 - h1 / 2, center_y2 - h2 / 2)
    c_b = torch.max(center_y1 + h1 / 2, center_y2 + h2 / 2)

    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    c_diag = torch.clamp((c_r - c_l), min=0) ** 2 + torch.clamp((c_b - c_t), min=0) ** 2

    union = area1 + area2 - inter_area
    u = (inter_diag) / c_diag
    iou = inter_area / union
    dious = iou - u
    dious = torch.clamp(dious, min=-1.0, max=1.0)
    losses = 1 - dious
    return losses.mean()


class IOULoss(nn.Module):
    def __init__(self, loc_loss_type):
        super(IOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
        target_area = (target_left + target_right) * (target_top + target_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        # g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        # g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        # ac_uion = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        # gious = ious - (ac_uion - area_union) / ac_uion

        if self.loc_loss_type == 'iou':
            losses = -torch.log(ious)
        # elif self.loc_loss_type == 'linear_iou':
        #     losses = 1 - ious
        # elif self.loc_loss_type == 'giou':
        #     losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum(), ious
        else:
            assert losses.numel() != 0
            return losses.mean()


linear_iou = IOULoss(loc_loss_type='iou')
