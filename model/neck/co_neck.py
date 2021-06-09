import torch.nn as nn


class UCR(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7):
        """
        UCR Block
        """
        super(UCR, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.center_size = center_size

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            l = (x.size(3) - self.center_size) // 2
            r = l + self.center_size
            x = x[:, :, l:r, l:r]
        return x


class Adjust(nn.Module):
    """
    RPN neck
    """
    def __init__(self, in_channels, out_channels, center_size=7):
        super(Adjust, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.center_size = center_size

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            l = (x.size(3) - self.center_size) // 2
            r = l + self.center_size
            x = x[:, :, l:r, l:r]
        return x


class ManNeck(nn.Module):
    def __init__(self, in_channels, out_channels, relu=True, center_size=7):
        """
        MAN neck
        """
        super(ManNeck, self).__init__()
        if relu:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.center_size = center_size

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            l = (x.size(3) - self.center_size) // 2
            r = l + self.center_size
            x = x[:, :, l:r, l:r]
        return x
