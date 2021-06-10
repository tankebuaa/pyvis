import torch
import torch.nn.functional as F
from model.base_builder import BaseBuilder
from model.backbone.efficientnet import EfficientNetB0
import model.backbone.resnet_atrous as res_atr
from model.neck.co_neck import UCR, Adjust
from model.head.co_head import ManHead, CoDWRPN
from model.loss.rpn_loss import select_cross_entropy_loss, weight_l1_loss


class CoModelBuilder(BaseBuilder):
    def __init__(self, backbone='resnet50c3'):
        super(CoModelBuilder, self).__init__()
        self.backbone_type = backbone
        if self.backbone_type is 'resnet50c3':
            self.backbone = res_atr.resnet50(used_layers=[2,],
                                            frozen_layers=['conv1', 'bn1', 'relu', 'maxpool', 'layer1', ])
            feat_channels = 512
        elif self.backbone_type is 'efficientnetb0':
            self.backbone = EfficientNetB0(pretrained=True, frozen_blocks=5)
            feat_channels = 320
        else:
            raise ValueError('backbone should be one of: resnet50c3 and efficientnetb0 !')

        self.man_neck = UCR(in_channels=feat_channels, out_channels=1024, center_size=7)
        self.man_head = ManHead()

        self.corpn_neck = Adjust(in_channels=feat_channels, out_channels=256, center_size=7)
        self.corpn_head = CoDWRPN(anchor_num=5, in_channels=256, out_channels=256)

        self.ERASETH = 0.9

    def forward(self, data):
        """
        only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()
        label_fc = data['label_fc'].cuda()
        label_fc_weight = data['label_fc_weight'].cuda()

        z = self.backbone(template)
        x = self.backbone(search)

        zf_man = self.man_neck(z)
        xf_man = self.man_neck(x)
        dw_cls_man, cls_man = self.man_head(zf_man, xf_man)

        MAM = self.get_match_active_map(dw_cls_man.detach(), cls_man, xf_man.detach())#2label_fc_delta
        pos = torch.ge(MAM, self.ERASETH)
        mask = torch.ones(x.size(0), x.size(2), x.size(3)).cuda()
        mask[pos.data] = 0.0
        mask = torch.unsqueeze(mask, dim=1)

        zf = self.corpn_neck(z)
        xf = self.corpn_neck(x)
        co_xf = xf * mask
        cls_rpn1, cls_rpn2, loc = self.corpn_head(zf, xf, co_xf)

        # loss
        loss_cls_man = F.binary_cross_entropy_with_logits(cls_man, label_fc, weight=label_fc_weight)
        loss_cls_rpn1 = select_cross_entropy_loss(cls_rpn1, label_cls)
        lss_cls_rpn2 = select_cross_entropy_loss(cls_rpn2, label_cls)
        loss_loc = weight_l1_loss(loc, label_loc, label_loc_weight)

        # get total loss
        outputs = {}
        outputs['total_loss'] = loss_cls_man + (loss_cls_rpn1 + lss_cls_rpn2) / 2.0 + 1.2 * loss_loc
        outputs['man_loss'] = loss_cls_man
        outputs['ori_cls_loss'] = lss_cls_rpn2
        outputs['co_cls_loss'] = loss_cls_rpn1
        outputs['loc_loss'] = loss_loc
        return outputs

    def get_match_active_map(self, responses, cls, feature_map):
        """
            Compute the matching activation map, refer as MAM
        """
        b, c, h, w = responses.shape
        cls = cls.view(b, -1)
        _, maxi = torch.max(cls, dim=1)
        peaks = torch.stack([maxi % h, maxi / h], dim=1).float().cuda()
        # peaks = torch.Tensor([[h // 2, w // 2]]).cuda() + cls
        MAM = torch.zeros([b, feature_map.size(-2), feature_map.size(-1)]).cuda()
        for bi in range(b):
            # compute bbox center loc
            peak = peaks[bi, :]
            # compute response peak loc per channel
            response = responses[bi].view(c, -1)
            _, maxi = torch.max(response, dim=1)
            loc = torch.stack([maxi % h, maxi / h]).float().cuda()
            active = torch.sum(torch.abs((loc - peak.unsqueeze(1))), dim=0) <= 2
            num_chs = torch.sum(active)
            if num_chs > 0:
                # # --------------------------
                atten_maps = feature_map[bi][active]
                ch_mins, _ = torch.min(atten_maps.view(num_chs, -1), dim=-1, keepdim=True)
                ch_maxs, _ = torch.max(atten_maps.view(num_chs, -1), dim=-1, keepdim=True)
                atten_normed = torch.div(atten_maps.view(num_chs, -1) - ch_mins,
                                         ch_maxs - ch_mins)
                atten_normed = atten_normed.view(atten_maps.shape)
                cmap = torch.sum(atten_normed, dim=0)
                # #---------------------------
                val_min = torch.min(cmap)
                val_max = torch.max(cmap)
                MAM[bi, :, :] = (cmap - val_min) / (val_max - val_min)
        return MAM

    def template(self, template):
        z = self.backbone(template)
        self.fa_z = self.man_neck(z)
        self.zf = self.corpn_neck(z)

    def track(self, search):
        x = self.backbone(search)
        fa_x = self.man_neck(x)
        xf = self.corpn_neck(x)

        # get man response map
        dw_cls_man, cls_man = self.man_head(self.fa_z, fa_x)
        response = cls_man.squeeze()

        MAM = self.get_match_active_map(dw_cls_man, cls_man, fa_x)
        pos = torch.ge(MAM, self.ERASETH)
        mask = torch.ones(x.size(0), x.size(2), x.size(3)).cuda()
        mask[pos.data] = 0.0
        mask = torch.unsqueeze(mask, dim=1)
        co_xf = xf * mask

        # get CoRPN outputs
        cls_rpn1, cls_rpn2, loc = self.corpn_head(self.zf, xf, co_xf)

        return {
            'cls1': cls_rpn1,
            'cls2': cls_rpn2,
            'loc': loc,
            'response': response,
        }

    def flops_forward(self, search): # flops_
        """
        only used in flops counting
        """
        template = torch.randn(1, 3, 127, 127).cuda()
        z = self.backbone(template)
        x = self.backbone(search)

        zf_man = self.man_neck(z)
        xf_man = self.man_neck(x)
        dw_cls_man, cls_man = self.man_head(zf_man, xf_man)

        MAM = self.get_match_active_map(dw_cls_man.detach(), cls_man, xf_man.detach())#2label_fc_delta
        pos = torch.ge(MAM, self.ERASETH)
        mask = torch.ones(x.size(0), x.size(2), x.size(3)).cuda()
        mask[pos.data] = 0.0
        mask = torch.unsqueeze(mask, dim=1)

        zf = self.corpn_neck(z)
        xf = self.corpn_neck(x)
        co_xf = xf * mask
        cls_rpn1, cls_rpn2, loc = self.corpn_head(zf, xf, co_xf)
        return cls_rpn1, cls_rpn2, loc
