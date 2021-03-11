import torch.nn as nn
from .layers import xcorr_depthwise


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


class RegHead(nn.Module):
    """
    REG
    """
    def __init__(self, in_channels, num_stage=2):
        super(RegHead, self).__init__()
        # reg
        # extract_regfeat = [AdjustLayer(num_stage * in_channels, in_channels, center_size=14),]
        # self.add_module('extract_regfeat', nn.Sequential(*extract_regfeat))
        extract_regfeat_z = [nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0, bias=False),
                             nn.BatchNorm2d(in_channels),
                             nn.ReLU(inplace=True)]
        self.add_module('extract_regfeat_z', nn.Sequential(*extract_regfeat_z))

        extract_regfeat_x = [nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0, bias=False),
                             nn.BatchNorm2d(in_channels),
                             nn.ReLU(inplace=True)]
        self.add_module('extract_regfeat_x', nn.Sequential(*extract_regfeat_x))

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

    def forward(self, zf, xf):
        # zf = self.extract_regfeat(z)
        # xf = self.extract_regfeat(x)
        zf_reg = self.extract_regfeat_z(zf)
        xf_reg = self.extract_regfeat_x(xf)
        dwfeat = xcorr_depthwise(xf_reg, zf_reg)
        bbox_reg = self.bbox_pred(dwfeat).exp()
        return bbox_reg