import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cv2

import torch
import torch.nn.functional as F
import model.backbone.resnet_atrous as res_atr
from model.neck import NeckAllLayer
from model.head import IAMFHead
from model.loss import select_cross_entropy_loss, select_iou_loss#, sigmoid_focal_loss
from model.base_builder import BaseBuilder


class ModelBuilder(BaseBuilder):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        # self.backbone = res.resnet50(output_layers=['layer2'], pretrained=True,
        #                              frozen_layers=['conv1', 'bn1', 'relu', 'maxpool', ])
        self.backbone = res_atr.resnet50(used_layers=[2, 3, 4, ],
                                         frozen_layers=['conv1', 'bn1', 'relu', 'maxpool', 'layer1',])
                                                        #  'layer2', 'layer3', 'layer4',])

        # build neck
        self.neck = NeckAllLayer(in_channels=[512, 1024, 2048], out_channels=[256, 256, 256]) #

        # build head
        self.head = IAMFHead(256, num_convs=3, num_stages=3)

    def forward(self, data):
        # only used for training
        template = data['template'].cuda()
        search = data['search'].cuda()
        template_box = data['template_box'].cuda()
        # search_box = data['search_box'].cuda()
        cls_label = data['cls_label'].cuda()
        reg_label = data['reg_label'].cuda()
        centerness_label = data['center_label'].cuda()

        z = self.backbone(template)
        x = self.backbone(search)

        # if torch.sum(torch.isnan(z[0].detach().flatten())).item() > 0:
        #     print("isnan")

        feat = self.neck(z, x, template_box - 15)

        cls_s, centerness_s, reg_s, iou_s = self.head(feat)

        flatten_cls_label = cls_label.view(-1, 1)  # (num_all)
        flatten_reg_label = reg_label.view(-1, 4)  # (num_all, 4)
        flatten_centerness_label = centerness_label.view(-1)  # (num_all)
        pos_inds = torch.nonzero(flatten_cls_label.reshape(-1)).squeeze(1)
        num_pos = len(pos_inds)
        pos_centerness_label = flatten_centerness_label[pos_inds]
        pos_reg_label = flatten_reg_label[pos_inds]

        loss_cls = []
        loss_centerness = []
        loss_reg = []
        loss_iou = []
        for cls, centerness, reg, iou in zip(cls_s, centerness_s, reg_s, iou_s):
            flatten_centerness_pred = centerness.permute(0, 2, 3, 1).reshape(-1)
            flatten_reg_pred = reg.permute(0, 2, 3, 1).reshape(-1, 4)
            flatten_iou_pred = iou.permute(0, 2, 3, 1).reshape(-1)
            pos_centerness_pred = flatten_centerness_pred[pos_inds]
            pos_reg_pred = flatten_reg_pred[pos_inds]
            pos_iou_pred = flatten_iou_pred[pos_inds]
            # print(pos_reg_label[0,0].item(), pos_reg_pred[0,0].item())
            # flatten_cls_pred = cls.permute(0, 2, 3, 1).reshape(-1, 1)
            # loss_cls = sigmoid_focal_loss(flatten_cls_pred, flatten_cls_label, 2.0, 0.25, 'mean') # avoid num_pos is 0
            loss_cls.append(select_cross_entropy_loss(cls, flatten_cls_label))

            if num_pos > 0:
                loss_centerness.append(F.binary_cross_entropy_with_logits(
                    pos_centerness_pred, pos_centerness_label, reduction='mean')[None])
                # centerness weighted iou loss
                _loss_reg, ious_target = select_iou_loss(pos_reg_pred, pos_reg_label, pos_centerness_label)
                loss_reg.append(_loss_reg)
                loss_iou.append(F.binary_cross_entropy_with_logits(pos_iou_pred, ious_target, reduction='mean'))
            else:
                loss_reg.append(pos_reg_pred.sum())
                loss_centerness.append(pos_centerness_pred.sum())
                loss_iou.append(pos_iou_pred.sum())

        def avg(lst):
            return sum(lst) / len(lst)

        outputs = {'total_loss': 1.0 * avg(loss_cls) + 0.5 * avg(loss_centerness) +
                                 1.2 * avg(loss_reg) + 0.5 * avg(loss_iou),
                   'cls_loss': avg(loss_cls),
                   'centerness_loss': avg(loss_centerness),
                   'box_loss': avg(loss_reg),
                   'iou_loss': avg(loss_iou),}
        return outputs

    def template(self, z, bbox):
        z = self.backbone(z)
        self.kernels = self.neck.get_kernel(z, bbox-15)

    def track(self, x):
        x = self.backbone(x)
        features = self.neck.get_feature(self.kernels, x) #size:[3]
        cls_s, centerness_s, reg_s, iou_s = self.head(features)
        return cls_s, centerness_s, reg_s, iou_s
