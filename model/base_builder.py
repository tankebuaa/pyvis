import torch.nn as nn


class BaseBuilder(nn.Module):
    def forward(self, data):
        raise NotImplementedError