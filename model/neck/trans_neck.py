import torch
from torch import nn
import torch.nn.functional as F


def dw_xcorr(x, kernel):
    """Padded-Depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch * channel, x.size(2), x.size(3))
    kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch * channel)#, padding=kernel.size(2)//2)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


class TransNeck(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(TransNeck, self).__init__()
        self.input_proj_search = nn.Conv2d(in_dim, hidden_dim, kernel_size=1)
        self.input_proj_kernel = nn.Conv2d(in_dim, hidden_dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(hidden_dim)

    def forward(self, search, kernel, box):
        search = self.input_proj_search(search)
        kernel = self.input_proj_kernel(kernel)
        query = dw_xcorr(search, kernel)
        query_embd = self.bn(query)
        return search, query_embd

    def convert_to_roi_format(self, boxes):
        concat_boxes = self.cat([b.bbox for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = self.cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def cat(self, tensors, dim=0):
        """
        Efficient version of torch.cat that avoids a copy if there is only a single element in a list
        """
        assert isinstance(tensors, (list, tuple))
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim)
