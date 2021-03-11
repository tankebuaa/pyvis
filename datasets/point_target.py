import torch
import numpy as np


class PointTarget:
    """
    This class generate points map from the prediction feature map .
    """
    def __init__(self, stride, size, center_sampling_radius=0.6):
        self.stride = stride
        self.size = size
        self.center_sampling_radius = center_sampling_radius / 2.0
        offset = 2 * stride - 1
        self.locations = self.generate_points(self.stride, self.size, offset)
        self.pos_num = size **2 #16
        self.total_num = size**2 # 64

    def generate_points(self, stride, size, offset):
        shifts_x = torch.arange(0, size * stride, step=stride, dtype=torch.float32)
        shifts_y = torch.arange(0, size * stride, step=stride, dtype=torch.float32)
        y, x = torch.meshgrid(shifts_x, shifts_y)
        shift_x = x.reshape(-1)
        shift_y = y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + offset
        return locations

    def __call__(self, bbox, neg_sample=False):
        "bbox:[x,y,w,h]"
        xs, ys = self.locations[:, 0], self.locations[:, 1]

        l = xs - bbox[0]
        t = ys - bbox[1]
        r = -xs + bbox[2]
        b = -ys + bbox[3]

        reg_target = torch.stack([l, t, r, b], dim=-1)

        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num

        is_in_boxes = self.get_sample_region(
            bbox,
            self.stride,
            xs, ys,
            radius=self.center_sampling_radius
        )
        is_out_boxes = reg_target.min(dim=1)[0] > 0

        pos, posnum = select(np.where(is_in_boxes == True), self.pos_num)
        neg, negnum = select(np.where(is_out_boxes == False), self.total_num - posnum)

        cls = -1 * np.ones((self.size * self.size, 1), dtype=np.int64)
        if neg_sample:
            cls[pos, 0] = 0
        else:
            cls[pos, 0] = 1
        cls[neg, 0] = 0

        centerness = self.compute_centerness_targets(reg_target, neg)
        return cls, reg_target, centerness

    def get_sample_region(self, gt, stride, gt_xs, gt_ys, radius=0.6):
        center_x = (gt[0] +gt[2]) / 2
        center_y = (gt[1] +gt[3]) / 2
        center_gt = np.zeros(len(gt))

        width = gt[2] - gt[0]
        height = gt[3] - gt[1]

        # no gt
        if center_x == 0:
            return np.zeros(gt_xs.shape, dtype=np.uint8)

        stride_x = width * radius
        stride_y = height * radius
        xmin = center_x - stride_x
        ymin = center_y - stride_y
        xmax = center_x + stride_x
        ymax = center_y + stride_y
        center_gt[0] = xmin if xmin > gt[0] else gt[0]
        center_gt[1] = ymin if ymin > gt[1] else gt[1]
        center_gt[2] = gt[2] if xmax > gt[2] else xmax
        center_gt[3] = gt[3] if ymax > gt[3] else ymax

        left = gt_xs[:] - center_gt[0]
        right = center_gt[2] - gt_xs[:]
        top = gt_ys[:] - center_gt[1]
        bottom = center_gt[3] - gt_ys[:]
        center_bbox = torch.stack((left, top, right, bottom), dim=0)
        inside_gt_bbox_mask = center_bbox.min(0)[0] > 0
        return inside_gt_bbox_mask

    def compute_centerness_targets(self, reg_targets, neg):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        centerness[neg] = 0
        return torch.sqrt(centerness)
