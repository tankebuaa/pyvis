import torch
import torch.nn as nn
import torch.nn.functional as F
from .iou_net import IoUNet
from config import basi_cfg as cfg


class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=14):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.center_size = center_size

    def forward(self, x):
        if x.size(3) < 20:
            l = (x.size(3) - self.center_size) // 2
            r = l + self.center_size
            x = x[:, :, l:r, l:r]
        x = self.downsample(x)
        return x


class BasiHead(nn.Module):
    """
    CLS(),REG,IOU
    """
    def __init__(self, in_channels, num_stage=2):
        super(BasiHead, self).__init__()
        # cls
        extract_clsfeat = [AdjustLayer(num_stage * in_channels, in_channels, center_size=14),
                           nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0, bias=False),
                           nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True)]
        self.add_module('extract_clsfeat', nn.Sequential(*extract_clsfeat))

        # reg
        extract_regfeat = [AdjustLayer(num_stage * in_channels, in_channels, center_size=14),]
        self.add_module('extract_regfeat', nn.Sequential(*extract_regfeat))
        extract_regfeat_z = [nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0, bias=False),
                             nn.BatchNorm2d(in_channels),
                             nn.ReLU(inplace=True)]
        self.add_module('extract_regfeat_z', nn.Sequential(*extract_regfeat_z))

        extract_regfeat_x = [nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0, bias=False),
                             nn.BatchNorm2d(in_channels),
                             nn.ReLU(inplace=True)]
        self.add_module('extract_regfeat_x', nn.Sequential(*extract_regfeat_x))

        # cls, reg -> endpoint
        self.cls_logits = nn.BatchNorm2d(1)
        self.bbox_pred = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(in_channels),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1),)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_()
                m.bias.data.zero_()

        # IoUNet
        self.iounet = IoUNet(input_dim=num_stage * in_channels, pred_input_dim=in_channels, pred_inter_dim=in_channels)

    def forward(self, z, x, train_bb, test_proposals):
        zf_cls = self.extract_clsfeat(z)
        xf_cls = self.extract_clsfeat(x)

        # spatial_atten = torch.softmax(torch.sum(zf_cls, dim=1).reshape(zf_cls.size(0), -1), dim=1)\
        #     .reshape(zf_cls.size(0), 1, zf_cls.size(2), zf_cls.size(3))
        cls = self.xcorr_fast(xf_cls, zf_cls)  # * spatial_atten)# * 0.001
        cls = self.cls_logits(cls)

        zf = self.extract_regfeat(z)
        xf = self.extract_regfeat(x)
        zf_reg = self.extract_regfeat_z(zf)
        xf_reg = self.extract_regfeat_x(xf)
        dwfeat = self.xcorr_depthwise(xf_reg, zf_reg)
        bbox_reg = self.bbox_pred(dwfeat).exp()

        # IoUNet
        iou_pred = self.iounet(z, x, train_bb.unsqueeze(0), test_proposals.unsqueeze(0))

        return cls, bbox_reg, iou_pred

    def xcorr_fast(self, x, kernel):
        """group conv2d to calculate cross correlation, fast version
        """
        batch = kernel.size()[0]
        pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
        px = x.view(x.size()[0] // batch, -1, x.size()[2], x.size()[3])
        po = F.conv2d(px, pk, groups=batch)
        po = po.view(-1, 1, po.size()[2], po.size()[3])
        return po

    def xcorr_depthwise(self, x, kernel):
        """depthwise cross correlation
        """
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(-1, batch * channel, x.size(2), x.size(3))
        kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch * channel)
        out = out.view(-1, channel, out.size(2), out.size(3))
        return out

    def template(self, z, bbox, x):
        # cls
        self.zf_cls = self.extract_clsfeat(z)
        # self.spatial_atten = torch.softmax(torch.sum(self.zf_cls, dim=1).reshape(1, -1), dim=1) \
        #     .reshape(1, 1, self.zf_cls.size(2), self.zf_cls.size(3))
        # TODO: background supress in template
        L = self.zf_cls.shape[-1]
        mh = min(torch.round(bbox[0,3] / 16).int() * 2, L)
        mw = min(torch.round(bbox[0,2] / 16).int() * 2, L)
        ph = (L - mh) // 2
        pw = (L - mw) // 2
        padding = torch.nn.ZeroPad2d(padding=(pw, pw, ph, ph))
        mask = padding(torch.ones([1, 1, mh, mw]).cuda())
        self.clf_kernel = self.zf_cls * mask# * self.spatial_atten

        # target-aware vector
        dscores = self.xcorr_depthwise(self.extract_clsfeat(x), self.clf_kernel)
        dscore = dscores.view(dscores.shape[1], -1)
        _, maxi = torch.max(dscore, dim=1)
        loc = torch.stack([maxi % cfg.SCORE_SIZE, maxi / cfg.SCORE_SIZE])
        center = torch.Tensor([[cfg.SCORE_SIZE // 2], [cfg.SCORE_SIZE // 2]]).cuda()
        self.target_weight = 1.0 * (torch.sum(abs(loc - center),dim=0) <= 2)

        # cmap = torch.sum(self.zf_cls * target_weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), dim=1)
        # saliency_maps_flat = cmap.view(cmap.size(0), -1)
        # ch_mins, _ = torch.min(saliency_maps_flat, dim=-1, keepdim=True)
        # ch_maxs, _ = torch.max(saliency_maps_flat, dim=-1, keepdim=True)
        # salient_map = torch.div(saliency_maps_flat - ch_mins, ch_maxs - ch_mins + 1e-10).view(cmap.shape)
        # pos = torch.ge(salient_map, 0.4)
        # mask = torch.ones(self.zf_cls.size(0), self.zf_cls.size(2), self.zf_cls.size(3)).cuda()
        # mask[pos.data] = 0.0
        # self.clf_kernel = self.clf_kernel * mask

        # baground aware vector
        # self.clf_kernel_exlude = self.zf_cls - self.clf_kernel
        self.salient_weight = torch.zeros([1, 256, 1, 1]).cuda()

        # regression and iou
        self.zf_reg = self.extract_regfeat_z(self.extract_regfeat(z))

        # IoUNet
        self.iou_modulation = self.iounet.get_modulation(z, bbox)

    def update_salient(self, weight):
        x = torch.sum(weight, dim=0)
        # print((torch.sum(x * self.target_weight)/torch.sum(self.target_weight)).cpu().numpy())
        # if torch.sum(x * self.target_weight)/torch.sum(self.target_weight) > 0.30:#F.cosine_similarity(x, self.target_weight, dim=0) > 0.40:
        #     return
        x = x * (1.0 - self.target_weight)
        w = (1 - cfg.ALPHA) * self.salient_weight + cfg.ALPHA * x.view(1, -1, 1, 1)#/(torch.sum(x,dtype=torch.float32) + 1e-10)
        self.salient_weight = w / (torch.max(w) + 1e-10)

    def track(self, x):
        xf_cls = self.extract_clsfeat(x)
        # fast version
        scores = self.xcorr_fast(xf_cls, self.clf_kernel)
        cls = self.cls_logits(scores).squeeze()

        xf_reg = self.extract_regfeat_x(self.extract_regfeat(x))
        dwfeat = self.xcorr_depthwise(xf_reg, self.zf_reg)
        bbox_reg = self.bbox_pred(dwfeat).exp().squeeze()

        return cls, bbox_reg, None

    def ba_track(self, x, visdom=None):
        xf_cls = self.extract_clsfeat(x)
        # original map
        dscores = self.xcorr_depthwise(xf_cls, self.clf_kernel)
        # scores_raw = torch.sum(dscores, dim=1, keepdim=True)

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
        scores_co = self.xcorr_fast(xf_cls_co, self.clf_kernel)

        # debug
        if visdom is not None:
            visdom.register(salient_map, 'heatmap', 3, 'salient_map')
            visdom.register(torch.sum(dscores, dim=1, keepdim=True), 'heatmap', 3, 'raw_map')
            visdom.register(scores_co, 'heatmap', 3, 'co_map')

        # regression
        xf_reg = self.extract_regfeat_x(self.extract_regfeat(x))
        dwfeat = self.xcorr_depthwise(xf_reg, self.zf_reg)
        bbox_reg = self.bbox_pred(dwfeat).exp().squeeze()

        return dscores.squeeze(), scores_co.squeeze(), bbox_reg,