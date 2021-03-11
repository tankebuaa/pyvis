import torch.nn as nn
from .layers import AdjustLayer, xcorr_depthwise, xcorr_fast


class ClsHead(nn.Module):
    """
    CLS(),REG,IOU
    """
    def __init__(self, in_channels, num_stage=2):
        super(ClsHead, self).__init__()
        # cls
        extract_clsfeat = [nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0, bias=False),
                             nn.BatchNorm2d(in_channels),
                             nn.ReLU(inplace=True)]
        self.add_module('extract_clsfeat', nn.Sequential(*extract_clsfeat))

        # cls, reg -> endpoint
        self.cls_logits = nn.BatchNorm2d(1)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_()
                m.bias.data.zero_()

    def forward(self, z, x):
        zf_cls = self.extract_clsfeat(z)
        xf_cls = self.extract_clsfeat(x)

        # spatial_atten = torch.softmax(torch.sum(zf_cls, dim=1).reshape(zf_cls.size(0), -1), dim=1)\
        #     .reshape(zf_cls.size(0), 1, zf_cls.size(2), zf_cls.size(3))
        cls = xcorr_fast(xf_cls, zf_cls)  # * spatial_atten)# * 0.001
        cls = self.cls_logits(cls)
        return cls