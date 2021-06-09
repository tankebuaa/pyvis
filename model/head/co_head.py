import torch
import torch.nn as nn
import torch.nn.functional as F


def xcorr_depthwise(x, kernel):
    """
    Depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch * channel, x.size(2), x.size(3))
    kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch * channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError


class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1)
                )

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = xcorr_depthwise(search, kernel)
        out = self.head(feature)
        return out


class DWRPN(RPN):
    def __init__(self, anchor_num=5, in_channels=256, out_channels=256):
        super(DWRPN, self).__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, 2 * anchor_num)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4 * anchor_num)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc


class CoDWRPN(nn.Module):
    def __init__(self, anchor_num=5, in_channels=256, out_channels=256):
        super(CoDWRPN, self).__init__()
        self.co_cls = DepthwiseXCorr(in_channels, out_channels, 2 * anchor_num, kernel_size=1)
        self.cls = DepthwiseXCorr(in_channels, out_channels, 2 * anchor_num, kernel_size=3)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4 * anchor_num, kernel_size=3)

    def forward(self, z_f, x_f, co_x_f):
        co_cls = self.co_cls(z_f, co_x_f)
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return co_cls, cls, loc


class ManHead(nn.Module):
    def __init__(self):
        super(ManHead, self).__init__()
        self.BN = nn.BatchNorm2d(1)

    def forward(self, z, x):
        dw_cls = xcorr_depthwise(x, z)
        cls = torch.sum(dw_cls, dim=1, keepdim=True)
        cls = self.BN(cls)
        return dw_cls, cls

    def track(self, kernel, x):
        """
        group conv2d to calculate cross correlation, fast version
        """
        cls = F.conv2d(x, kernel)
        cls = self.BN(cls)
        return cls