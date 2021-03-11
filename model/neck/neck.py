import torch
import torch.nn as nn
import torch.nn.functional as F
from external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from torchvision.ops import roi_align


def xcorr_depthwise(x, kernel):
    """Depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch * channel, x.size(2), x.size(3))
    kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch * channel, padding=kernel.size(2)//2)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, kernel_size=3):
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

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = xcorr_depthwise(search, kernel)
        return feature


class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            )
        self.center_size = center_size

    def forward(self, x):
        x = self.downsample(x)
        # if x.size(3) < 20:
        #     l = (x.size(3) - self.center_size) // 2
        #     r = l + self.center_size
        #     x = x[:, :, l:r, l:r]
        return x


class MatchCatNet(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3, center_size=7):
        super(MatchCatNet, self).__init__()
        self.prpool_z = PrRoIPool2D(1, 1, 1 / 8)
        self.conv_diff = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.center_size = center_size

        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        # self.downsample = nn.Sequential(
        #     nn.Conv2d(2 * hidden, out_channels, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        # )

    def forward(self, kernel, search, template_box):
        # Diff OP: templa te_box:[x1, y1, x2, y2]
        kernel_diff = self.conv_diff(kernel)
        search_diff = self.conv_diff(search)

        batch_size = template_box.shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1).cuda()
        template_box = template_box.clone()
        roi_z = torch.cat((batch_index, template_box), dim=1)

        kernel_vector = self.prpool_z(kernel_diff, roi_z)

        # Corr OP:
        l = (kernel.size(3) - self.center_size) // 2
        r = l + self.center_size
        kernel = kernel[:, :, l:r, l:r]
        kernel_corr = self.conv_kernel(kernel)
        search_corr = self.conv_search(search)

        feature_corr = xcorr_depthwise(search_corr, kernel_corr)

        # Concatenate
        feature = torch.cat((torch.abs(search_diff - kernel_vector), search_diff, feature_corr), dim=1) #
        # return self.downsample(feature)
        return feature


    def get_kernel(self, z, template_box):
        kernel_diff = self.conv_diff(z)
        batch_size = 1
        batch_index = torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1).cuda()
        template_box = template_box.clone()
        roi_z = torch.cat((batch_index, template_box), dim=1)

        kernel_vector = self.prpool_z(kernel_diff, roi_z)
        # kernel_vector = roi_align(kernel_diff, roi_z, (1,1), 1/8)

        l = (z.size(3) - self.center_size) // 2
        r = l + self.center_size
        # crop_z = z[:, :, l:r, l:r]
        kernel_corr = self.conv_kernel(z[:, :, l:r, l:r])
        return kernel_vector, kernel_corr

    def get_feature(self, kernels, x):
        kernel_vector = kernels[0]
        kernel_corr = kernels[1]
        search_diff = self.conv_diff(x)
        search_corr = self.conv_search(x)
        feature_corr = xcorr_depthwise(search_corr, kernel_corr)
        feature = torch.cat((torch.abs(search_diff - kernel_vector), search_diff, feature_corr), dim=1) #
        # return self.downsample(feature)
        return feature


class NeckAllLayer(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7):
        super(NeckAllLayer, self).__init__()
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = AdjustLayer(in_channels[0],
                                          out_channels[0],
                                          center_size)
            self.matchcat = MatchCatNet(in_channels=out_channels[0],
                                        hidden=out_channels[0],
                                        out_channels=out_channels[0])
        else:
            for i in range(self.num):
                self.add_module('downsample'+str(i+2),
                                AdjustLayer(in_channels[i],
                                            out_channels[i],
                                            center_size))
                self.add_module('matchcat'+str(i+2),
                                MatchCatNet(in_channels=out_channels[0],
                                            hidden=out_channels[0],
                                            out_channels=out_channels[0]))



    def forward(self, z_features, x_features, bbox):
        feature = []
        if self.num == 1:
            kernel = self.downsample(z_features)
            search = self.downsample(x_features)
            feature.append(self.matchcat(kernel, search, bbox))
            return feature
        else:
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample'+str(i+2))
                mc_layer = getattr(self, 'matchcat'+str(i+2))
                kernel = adj_layer(z_features[i])
                search = adj_layer(x_features[i])
                feat = mc_layer(kernel, search, bbox)
                feature.append(feat)
            return feature

    def get_kernel(self, z_features, bbox):
        kernels = []
        if self.num == 1:
            kernel = self.downsample(z_features)
            kernels.append(self.matchcat.get_kernel(kernel, bbox))
            return kernels
        else:
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample'+str(i+2))
                mc_layer = getattr(self, 'matchcat'+str(i+2))
                kernel = adj_layer(z_features[i])
                feat = mc_layer.get_kernel(kernel, bbox)
                kernels.append(feat)
        return kernels

    def get_feature(self, kernels, x_features):
        features = []
        for i in range(self.num):
            adj_layer = getattr(self, 'downsample' + str(i + 2))
            mc_layer = getattr(self, 'matchcat' + str(i + 2))
            search = adj_layer(x_features[i])
            feat = mc_layer.get_feature(kernels[i], search)
            features.append(feat)
        return features