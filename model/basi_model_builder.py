import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_builder import BaseBuilder
import model.backbone.resnet as res
import model.backbone.alexnet as alex
from model.neck import FpnLayer
from model.head import ClsHead, RegHead, xcorr_fast, xcorr_depthwise
from model.loss.loss import weight_l2_loss
from model.loss.iou_loss import linear_iou
from model.head.iou_net import AtomIoUNet
from config import basi_cfg as cfg


class WeightLoss(nn.Module):
    def __init__(self, weight=1.0, factor=torch.ones(3)):
        super(WeightLoss, self).__init__()
        self.weights = nn.Parameter(torch.tensor([weight for _ in range(len(factor))], dtype=torch.float32), requires_grad=True)
        self.factor = factor.cuda()

    def forward(self, x):
        loss = torch.sum(x / (self.factor * self.weights**2)) + torch.log(torch.prod(1 + self.weights**2))
        return loss


class BsiModelBuilder(BaseBuilder):
    def __init__(self, backbone='resnet50'):
        super(BsiModelBuilder, self).__init__()

        if backbone is 'resnet50':
            # build resnet backbone, necks and heads
            layers= ['conv1', 'bn1', 'relu', 'maxpool', 'layer1']
            self.backbone = res.resnet50(output_layers=['layer2', 'layer3'], pretrained=True,
                                         frozen_layers=layers, activate_layers=['layer2', 'layer3'])  # 512, 1024, 256

            self.cls_neck = FpnLayer(512, 1024, 256)
            self.reg_neck = FpnLayer(512, 1024, 256)

            self.cls_head = ClsHead(256)
            self.reg_head = RegHead(256)

            self.iou_head = AtomIoUNet(input_dim=(512, 1024), pred_input_dim=(256, 256),
                                       pred_inter_dim=(256, 256))
        elif backbone is 'alexnet':
            # build google lenet, neck and head
            self.backbone = alex.alexnet(output_layers=['layer2', 'layer3'], pretrained=True,
                                         frozen_layers=['layer1'], activate_layers=['layer2', 'layer3'])  #192, 256

            self.cls_neck = FpnLayer(192, 256, 256, center_size=13)
            self.reg_neck = FpnLayer(192, 256, 256, center_size=13)

            self.cls_head = ClsHead(256)
            self.reg_head = RegHead(256)

            self.iou_head = AtomIoUNet(input_dim=(192, 256), pred_input_dim=(256, 256),
                                       pred_inter_dim=(256, 256))

        # multi-task weight
        self.lossweights = WeightLoss(weight=1.0, factor=torch.tensor([1,1,2],dtype=torch.float32))

    def forward(self, data):
        # only used for training
        template = data['template'].cuda()
        search = data['search'].cuda()
        template_box = data['template_box'].cuda()
        search_box = data['search_box'].cuda()
        test_proposals = data['test_proposals'].cuda()
        cls_label = data['cls_label'].cuda()
        cls_weights = data['cls_weights'].cuda()
        reg_label = data['reg_label'].cuda()
        reg_weights = data['reg_weights'].cuda()
        iou_label = data['proposal_iou'].cuda()
        weight_iou = data['iou_weight'].cuda()

        z = self.backbone(template)
        x = self.backbone(search)

        zf_cls = self.cls_neck(z['layer2'], z['layer3'])
        xf_cls = self.cls_neck(x['layer2'], x['layer3'])
        cls = self.cls_head(zf_cls, xf_cls)

        zf_reg = self.reg_neck(z['layer2'], z['layer3'])
        xf_reg = self.reg_neck(x['layer2'], x['layer3'])
        bbox_reg = self.reg_head(zf_reg, xf_reg)

        # iou_pred = self.iou_head(zf, xf, template_box.unsqueeze(0), test_proposals.unsqueeze(0))
        iou_pred = self.iou_head([z['layer2'], z['layer3']], [x['layer2'], x['layer3']],
                                 template_box.unsqueeze(0), test_proposals.unsqueeze(0))

        # zf = self.neck(z['layer2'], z['layer3'])
        # xf = self.neck(x['layer2'], x['layer3'])
        # cls, bbox_reg, iou_pred = self.head(zf, xf, template_box, test_proposals)

        loss_cls = F.binary_cross_entropy_with_logits(cls, cls_label, weight=cls_weights)

        flatten_reg_weights = reg_weights.view(-1, 1)  # (num_all)
        flatten_reg_label = reg_label.view(-1, 4)  # (num_all, 4)
        flatten_reg_pred = bbox_reg.permute(0, 2, 3, 1).reshape(-1, 4)
        pos_inds = flatten_reg_weights.squeeze().eq(1).nonzero().squeeze()
        num_pos = len(pos_inds)
        pos_reg_label = flatten_reg_label[pos_inds]
        pos_reg_pred = flatten_reg_pred[pos_inds]
        if num_pos > 0:
            loss_reg = linear_iou(pos_reg_pred, pos_reg_label)  # linear_iou   ciou
        else:
            print("all negative!")
            loss_reg = 0.0

        loss_iou = weight_l2_loss(iou_pred.reshape(-1, iou_pred.shape[-1]), iou_label, weight_iou)
        loss = self.lossweights(torch.cat([loss_cls.unsqueeze(0), loss_reg.unsqueeze(0), loss_iou.unsqueeze(0)]))
        # loss = loss_cls + 3 * loss_reg + 2 * loss_iou


        outputs = {'total_loss': loss,
                   'cls_loss': loss_cls,
                   'box_loss': loss_reg,
                   'iou_loss': loss_iou,
                   'cls_weight': self.lossweights.weights[0].detach(),
                   'box_weight': self.lossweights.weights[1].detach(),
                   'iou_weight': self.lossweights.weights[2].detach()}
        return outputs

    def template(self, img, bbox, search):
        # get template feature maps
        z = self.backbone(img)
        zf_cls = self.cls_neck(z['layer2'], z['layer3'])
        zf_cls = self.cls_head.extract_clsfeat(zf_cls)
        # TODO: background supress in template
        # self.spatial_atten = torch.softmax(torch.sum(self.zf_cls, dim=1).reshape(1, -1), dim=1) \
        #     .reshape(1, 1, self.zf_cls.size(2), self.zf_cls.size(3))
        if cfg.MASK:
            L = zf_cls.shape[-1]
            ex = L % 2
            mh = min(torch.round(bbox[0,3] / 16).int() * 2 + ex, L)
            mw = min(torch.round(bbox[0,2] / 16).int() * 2 + ex, L)
            ph = (L - mh) // 2
            pw = (L - mw) // 2
            padding = torch.nn.ZeroPad2d(padding=(pw, pw, ph, ph))
            mask = padding(torch.ones([1, 1, mh, mw]).cuda())
            self.clf_kernel = zf_cls * mask  # * self.spatial_atten
        else:
            self.clf_kernel = zf_cls

        """target-aware vector"""
        x = self.backbone(search)
        xf = self.cls_neck(x['layer2'], x['layer3'])
        dscores = xcorr_depthwise(self.cls_head.extract_clsfeat(xf), self.clf_kernel)
        dscore = dscores.view(dscores.shape[1], -1)
        _, maxi = torch.max(dscore, dim=1)
        loc = torch.stack([maxi % cfg.SCORE_SIZE, maxi / cfg.SCORE_SIZE])
        center = torch.Tensor([[cfg.SCORE_SIZE // 2], [cfg.SCORE_SIZE // 2]]).cuda()
        self.target_weight = 1.0 * (torch.sum(abs(loc - center),dim=0) <= 2)

        # baground aware vector
        self.salient_weight = torch.zeros([1, 256, 1, 1]).cuda()

        """regression"""
        zf_reg = self.reg_neck(z['layer2'], z['layer3'])
        self.zf_reg = self.reg_head.extract_regfeat_z(zf_reg)

        """IoUNet"""
        self.iou_modulation = self.iou_head.get_modulation([z['layer2'], z['layer3']], bbox)
        # self.iou_modulation = self.iou_head.get_modulation(zf, bbox)

    def track(self, img, visdom=None):
        # get search feature maps
        self.x = self.backbone(img)
        self.xf = self.neck(self.x['layer2'], self.x['layer3'])
        return self.head.ba_track(self.xf, visdom)

    def update_salient(self, weight):
        if weight is None:
            self.salient_weight = self.salient_weight * (1.0 - 0.5 * self.target_weight.view(1, -1, 1, 1))
            return
        x = torch.sum(weight, dim=0)
        x = x * (1.0 - 0.7 * self.target_weight)
        w = (1 - cfg.ALPHA) * self.salient_weight + cfg.ALPHA * x.view(1, -1, 1, 1)/(torch.sum(x, dtype=torch.float32) + 1e-16)
        self.salient_weight = w / (w.sum() + 1e-16)

    def ba_track(self, img, visdom=None):
        # get search feature maps
        self.x = self.backbone(img)
        self.xf_cls = self.cls_neck(self.x['layer2'], self.x['layer3'])

        # original corr map
        xf_cls = self.cls_head.extract_clsfeat(self.xf_cls)
        dscores = xcorr_depthwise(xf_cls, self.clf_kernel)

        # erased map
        cmap = torch.sum(xf_cls * self.salient_weight, dim=1)
        saliency_maps_flat = cmap.view(cmap.size(0), -1)
        ch_mins, _ = torch.min(saliency_maps_flat, dim=-1, keepdim=True)
        ch_maxs, _ = torch.max(saliency_maps_flat, dim=-1, keepdim=True)
        salient_map = torch.div(saliency_maps_flat - ch_mins, ch_maxs - ch_mins + 1e-10).view(cmap.shape)
        pos = torch.ge(salient_map, cfg.ERA_TH)
        mask = torch.ones(xf_cls.size(0), xf_cls.size(2), xf_cls.size(3)).cuda()
        mask[pos.data] = 0.0
        xf_cls_co = xf_cls * torch.unsqueeze(mask, dim=1)
        scores_co = xcorr_fast(xf_cls_co, self.clf_kernel)

        # debug
        if visdom is not None:
            visdom.register(F.interpolate(salient_map.unsqueeze(0),scale_factor=8,mode='bilinear').squeeze(), 'heatmap', 3, 'salient_map')

        # regression
        xf_reg = self.reg_head.extract_regfeat_x(self.reg_neck(self.x['layer2'], self.x['layer3']))
        dwfeat = xcorr_depthwise(xf_reg, self.zf_reg)
        bbox_reg = self.reg_head.bbox_pred(dwfeat).exp().squeeze()

        return dscores.squeeze(), scores_co.squeeze(), bbox_reg,

    def predict_iou(self, boxes):
        output_boxes = boxes.view(1, -1, 4)
        # IoUNet
        # self.iou_feat = self.iou_head.get_iou_feat(self.xf)
        # outputs = self.iou_head.predict_iou(self.iou_modulation, self.iou_feat, output_boxes) / 2 + 0.5

        # IoU_head
        feat1, feat2 = self.iou_head.get_iou_feat([self.x['layer2'], self.x['layer3']])
        self.iou_feat = [feat1.clone(), feat2.clone()]
        outputs = self.IoUhead.predict_iou(self.iou_modulation, [feat1, feat2], output_boxes).squeeze(0) / 2 + 0.5
        return outputs

    # ATOM
    def refine(self, boxes):
        feat1, feat2 = self.iou_head.get_iou_feat([self.x['layer2'], self.x['layer3']])
        self.iou_feat = [feat1.clone(), feat2.clone()]
        # self.iou_feat = self.iou_head.get_iou_feat(self.xf)
        output_boxes = boxes.view(1, -1, 4)
        step_length = 1.0
        box_refinement_step_decay = 0.5
        iter_num = 2
        with torch.set_grad_enabled(True):
            for i_ in range(iter_num):
                # forward pass
                bb_init = output_boxes.clone().detach()
                bb_init.requires_grad = True

                #ATOM
                outputs = self.iou_head.predict_iou(self.iou_modulation,
                                                       self.iou_feat,
                                                       bb_init)

                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]

                outputs.backward(gradient=torch.ones_like(outputs))

                # Update proposal
                output_boxes = bb_init + step_length * bb_init.grad * bb_init[:, :, 2:].repeat(1, 1, 2)
                output_boxes.detach_()

                step_length *= box_refinement_step_decay

        return output_boxes.view(-1,4), outputs.detach().squeeze(0)
