import torch
import torch.nn as nn


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


class FpnLayer(nn.Module):
    def __init__(self, in_channels_s2, in_chanels_s3, out_channels):
        super(FpnLayer, self).__init__()
        self.adj_s2 = nn.Sequential(
            nn.Conv2d(in_channels_s2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.adj_s3 = nn.Sequential(
            nn.Conv2d(in_chanels_s3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.adj_out = AdjustLayer(2 * out_channels, out_channels, center_size=14)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_()
                m.bias.data.zero_()

    def forward(self, x, y):
        feat_s2 = self.adj_s2(x)
        feat_s3 = self.adj_s3(y)
        return self.adj_out(self._upsample_cat(feat_s3, feat_s2))
        # return self._upsample_cat(feat_s3, feat_s2)

    def _upsample_cat(self, x, y):
        _, _, H, W = y.size()
        return torch.cat((nn.functional.interpolate(x, size=(H, W), mode='bilinear'), y), dim=1)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return nn.functional.interpolate(x, size=(H, W), mode='bilinear') + y


class PAFPN(nn.Module):
    def __init__(self, in_channels_s1, in_channels_s2, in_chanels_s3, out_channels):
        super(PAFPN, self).__init__()
        self.adj_s1 = nn.Sequential(
            nn.Conv2d(in_channels_s1, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.adj_s2 = nn.Sequential(
            nn.Conv2d(in_channels_s2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.adj_s3 = nn.Sequential(
            nn.Conv2d(in_chanels_s3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_()
                m.bias.data.zero_()

    def forward(self, x, y, z):
        feat_s1 = self.adj_s1(x)
        feat_s2 = self.adj_s2(y)
        feat_s3 = self.adj_s3(z)
        return self.sample_cat(feat_s3, feat_s2, feat_s1)

    def sample_cat(self, x, y, z):
        _, _, H, W = y.size()
        return torch.cat((nn.functional.interpolate(x, size=(H, W), mode='bilinear'), y, z), dim=1)

    def sample_add(self, x, y, z):
        _, _, H, W = y.size()
        return nn.functional.interpolate(x, size=(H, W), mode='bilinear') + y + z