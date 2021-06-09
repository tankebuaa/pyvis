import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIPool, RoIAlign


class projection_MLP(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=512, out_dim=512):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=False),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class ILHead(nn.Module):
    def __init__(self, indim, scale=1/8):
        super(ILHead, self).__init__()
        self.crop = [3, 13]
        self.pool = nn.AvgPool2d(kernel_size=10, stride=1, padding=0)
        self.projector = projection_MLP(indim, hidden_dim=indim, out_dim=indim)

    def forward(self, f1, box1, f2, box2):
        f1_crop = f1[:, :, self.crop[0]: self.crop[1], self.crop[0]: self.crop[1]]
        x1 = self.pool(f1_crop)
        x2 = self.pool(f2)
        z1 = self.projector(x1)
        z2 = self.projector(x2)
        return z1, z2

    def template(self, f1, box1):
        f1_crop = f1[:, :, self.crop[0]: self.crop[1], self.crop[0]: self.crop[1]]
        x1 = self.pool(f1_crop)
        self.z1 = self.projector(x1)

    def track(self, f2):
        x2 = self.pool(f2)
        z2 = self.projector(x2)
        s = F.cosine_similarity(self.z1, z2, dim=1)
        return s


## target level:resnet50-c4
class projection_TL(nn.Module):
    def __init__(self, in_dim, hidden_dim=1024, out_dim=1024):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1),
            # nn.BatchNorm2d(hidden_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x


class prediction_TL(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=256, out_dim=1024): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class TLHead(nn.Module):
    def __init__(self, indim, scale=1/16):
        super(TLHead, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=0) #RoIPool(output_size=1, spatial_scale=scale)
        self.projector = projection_TL(indim, hidden_dim=indim, out_dim=indim)
        self.predictor = prediction_TL(indim, hidden_dim=256, out_dim=indim)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.crop = [1, 6]

    def forward(self, f1, box1, f2, box2):
        f1_crop = f1[:, :, self.crop[0] : self.crop[1], self.crop[0] : self.crop[1]]
        # x1 = self.pool(f1, box1)
        # x2 = self.pool(f2, box2)
        z1 = self.projector(f1_crop)
        z2 = self.projector(f2)
        p1 = z1 # self.predictor(z1)
        p2 = z2 # self.predictor(z2)

        za1 = self.pool(z1)
        za2 = self.pool(z2)
        pa1 = self.pool(p1)
        pa2 = self.pool(p2)
        return z1, z2, p1, p2, za1, za2, pa1, pa2

    def template(self, f1, box1):
        f1_crop = f1[:, :, self.crop[0]: self.crop[1], self.crop[0]: self.crop[1]]
        self.z1 = self.projector(f1_crop)
        self.p1 = self.predictor(self.z1)
        self.za1 = self.pool(self.z1)
        self.pa1 = self.pool(self.p1)

    def track(self, f2):
        z2 = self.projector(f2)
        p2 = self.predictor(z2)
        za2 = self.pool(z2)
        pa2 = self.pool(p2)

        s1 = F.cosine_similarity(self.za1, pa2, dim=1)
        s2 = F.cosine_similarity(self.pa1, za2, dim=1)

        return s1, s2

# TODO: Part Level
class projection_PL(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=512):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1),
            # nn.BatchNorm2d(hidden_dim)
        )
        self.num_layers = 3

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x


class prediction_PL(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=256, out_dim=512): # bottleneck structure
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class PLHead(nn.Module):
    def __init__(self, indim, scale=1/8):
        super(PLHead, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=0)
        self.projector = projection_PL(indim, hidden_dim=indim, out_dim=indim)
        self.predictor = prediction_PL(indim, hidden_dim=indim//2, out_dim=indim)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, f1, box1, f2, box2):
        x1 = self.pool(f1, box1)
        # x2 = self.pool(f2, box2)
        z1 = self.projector(x1)
        z2 = self.projector(f2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return z1, z2, p1, p2

    def template(self, f1, box1):
        x1 = self.pool(f1, box1)
        self.z1 = self.projector(x1)
        self.p1 = self.predictor(self.z1)

    def track(self, f2):
        z2 = self.projector(f2)
        p2 = self.predictor(z2)
        s1 = F.cosine_similarity(self.z1, p2, dim=1)
        s2 = F.cosine_similarity(self.p1, z2, dim=1)
        return s1, s2