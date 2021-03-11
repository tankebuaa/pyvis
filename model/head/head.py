import torch
import torch.nn as nn
from utils.weight_init import normal_init, bias_init_with_prob


class Scale(nn.Module):

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        return x * self.scale


class IAMFHead(nn.Module):
    def __init__(self, in_channels, num_convs=3, num_stages=3):
        super(IAMFHead, self).__init__()
        cls_tower = []
        bbox_tower = []
        # down sample
        cls_tower.append(nn.Conv2d(3 * in_channels, in_channels, kernel_size=1))
        cls_tower.append(nn.GroupNorm(32, in_channels))
        cls_tower.append(nn.ReLU())
        bbox_tower.append(nn.Conv2d(3 * in_channels, in_channels, kernel_size=1))
        bbox_tower.append(nn.GroupNorm(32, in_channels))
        bbox_tower.append(nn.ReLU())
        for i in range(num_convs):
            cls_tower.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))

        self.cls_logits = nn.Conv2d(in_channels, 2, kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)
        self.iou_pred = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in range(num_stages)])

        # init
        for sub_model in [self.cls_tower, self.bbox_tower]:
            for m in sub_model.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.cls_logits, std=0.01, bias=bias_cls)
        normal_init(self.bbox_pred, std=0.01)
        normal_init(self.centerness, std=0.01)
        normal_init(self.iou_pred, std=0.01)

    def forward(self, x):
        #TODO: sigmoid_focal_loss, bce, iou_loss, bce
        cls = []
        cen = []
        bbox_reg = []
        iou = []
        for l, feat in enumerate(x):
            cls_tower = self.cls_tower(feat)
            box_tower = self.bbox_tower(feat)
            cls.append(self.cls_logits(cls_tower))
            cen.append(self.centerness(cls_tower))
            bbox_reg.append(self.scales[l](self.bbox_pred(box_tower)).exp())
            iou.append(self.iou_pred(box_tower))
        return cls, cen, bbox_reg, iou
        # def avg(lst):
        #     return sum(lst) / len(lst)
        #
        # return avg(cls), avg(cen), avg(bbox_reg), avg(iou)
