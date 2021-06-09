import torch
import torch.nn.functional as F
from model.base_builder import BaseBuilder
import model.backbone.resnet_atrous as res_atr
from model.neck.co_neck import ManNeck
from model.head.co_head import ManHead


class ManModelBuilder(BaseBuilder):
    def __init__(self, out_ch=1024, relu=True):
        super(ManModelBuilder, self).__init__()
        self.backbone = res_atr.resnet50(used_layers=[2,],
                                        frozen_layers=['conv1', 'bn1', 'relu', 'maxpool', 'layer1', ])
        feat_channels = 512
        self.man_neck = ManNeck(in_channels=feat_channels, out_channels=out_ch,
                                relu=relu, center_size=7)
        self.man_head = ManHead()

    def forward(self, data):
        """
        only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_fc = data['label_fc'].cuda()
        label_fc_weight = data['label_fc_weight'].cuda()

        z = self.backbone(template)
        x = self.backbone(search)

        zf_man = self.man_neck(z)
        xf_man = self.man_neck(x)
        _, cls_man = self.man_head(zf_man, xf_man)
        # loss
        loss_cls_man = F.binary_cross_entropy_with_logits(cls_man, label_fc, weight=label_fc_weight)
        # get total loss
        outputs = {}
        outputs['total_loss'] = loss_cls_man
        return outputs

    def template(self, template):
        z = self.backbone(template)
        self.fa_z = self.man_neck(z)

    def track(self, search):
        x = self.backbone(search)
        fa_x = self.man_neck(x)

        # get man response map
        cls_man = self.man_head.track(self.fa_z, fa_x)
        response = cls_man.squeeze()
        return {
            'response': response,
        }

    def visualize(self, search):
        x = self.backbone(search)
        fa_x = self.man_neck(x)
        dw_cls, cls_man = self.man_head(self.fa_z, fa_x)
        MAM = self.get_match_active_map(dw_cls, cls_man, fa_x)
        return MAM

    def get_match_active_map(self, responses, cls, feature_map):
        """
            Compute the matching activation map, refer as MAM
        """
        b, c, h, w = responses.size()
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

    def flops_forward(self, search):
        """
        only used in flops counting
        """
        template = torch.randn(1, 3, 127, 127).cuda()
        z = self.backbone(template)
        self.fa_z = self.man_neck(z)
        x = self.backbone(search)
        fa_x = self.man_neck(x)

        # get man response map
        cls_man = self.man_head.track(self.fa_z, fa_x)
        response = cls_man.squeeze()
        return {
            'response': response,
        }
