import torch
import numpy as np


class CenterReg:
    """
    This class generate center and regression maps from the prediction feature map .
    """
    def __init__(self, search_size, stride, size, cls_sampling_radius=4, box_sampling_radius=0.75):
        self.search_size = search_size
        self.stride = stride
        self.size = size
        self.center_sampling_radius = cls_sampling_radius / 2.0
        self.box_sampling_radius = box_sampling_radius / 2.0
        offset = (search_size - 1) / 2 - stride * (size // 2)
        self.locations = self.generate_points(self.stride, self.size, offset)
        self.pos_num = size ** 2  # 16
        self.total_num = 64  # 64

    def generate_points(self, stride, size, offset):
        shifts_x = torch.arange(0, size * stride, step=stride, dtype=torch.float32)
        shifts_y = torch.arange(0, size * stride, step=stride, dtype=torch.float32)
        y, x = torch.meshgrid(shifts_x, shifts_y)
        shift_x = x.reshape(-1)
        shift_y = y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + offset
        return locations

    def logistic_labels(self, x, y, r_pos, r_neg):
        dist = torch.abs(x) + torch.abs(y)  # block distance
        labels = np.where(dist <= r_pos,
                          np.ones_like(x),
                          np.where(dist < r_neg,
                                   -np.ones_like(x),
                                   np.zeros_like(x))).reshape(-1, 1)
        smooth_labels = torch.exp(-(x * x + y * y) / 3).reshape(-1, 1)
        return labels, smooth_labels


    def select(self, position, keep_num=16):
        num = position[0].shape[0]
        if num <= keep_num:
            return position, num
        slt = np.arange(num)
        np.random.shuffle(slt)
        slt = slt[:keep_num]
        return tuple(p[slt] for p in position), keep_num

    def __call__(self, bbox, neg_sample=False):
        """box label"""
        xs, ys = self.locations[:, 0], self.locations[:, 1]
        l = xs - bbox[0]
        t = ys - bbox[1]
        r = -xs + bbox[2]
        b = -ys + bbox[3]

        reg_target = torch.stack([l, t, r, b], dim=-1)
        is_valid_boxes = self.get_sample_region(
            bbox,
            self.stride,
            xs, ys,
            radius=self.box_sampling_radius
        )
        boxpos, boxnum = self.select(np.where(is_valid_boxes == True), self.total_num)
        box_weights = np.zeros((self.size * self.size, 1), dtype=np.float32)
        box_weights[boxpos] = 1

        """cls label"""
        delta = ((bbox[0] + bbox[2]) / 2.0 - self.search_size / 2.0) // self.stride, \
                ((bbox[1] + bbox[3]) / 2.0 - self.search_size / 2.0) // self.stride
        x = torch.arange(self.size) - (self.size // 2 + delta[0])
        y = torch.arange(self.size) - (self.size // 2 + delta[1])
        y, x = torch.meshgrid(y, x)
        r_pos = self.center_sampling_radius
        r_neg = 0
        is_in_boxes, labels = self.logistic_labels(x, y, r_pos, r_neg)
        # is_in_boxes = self.get_sample_region(
        #     bbox,
        #     self.stride,
        #     xs, ys,
        #     radius=self.center_sampling_radius
        # )
        pos, posnum = self.select(np.where(is_in_boxes == True), self.pos_num)
        neg, negnum = self.select(np.where(is_in_boxes == False), self.total_num - posnum)

        cls = -1 * np.ones((self.size * self.size, 1), dtype=np.float32)
        cls_weights = np.zeros_like(cls, dtype=np.float32)
        if neg_sample:
            cls[pos, 0] = 0
        else:
            cls[pos, 0] = labels[pos, 0] # 1
        cls[neg, 0] = labels[neg, 0] # 0
        cls_weights[pos] = 1
        cls_weights[neg] = 1

        return cls.reshape(1, self.size, self.size), reg_target,\
               cls_weights.reshape(1, self.size, self.size), box_weights.reshape(1, self.size, self.size)

    def get_sample_region(self, gt, stride, gt_xs, gt_ys, radius=2.0):
        center_x = (gt[0] +gt[2]) / 2
        center_y = (gt[1] +gt[3]) / 2
        center_gt = np.zeros(len(gt))

        # no gt
        if center_x == 0:
            return np.zeros(gt_xs.shape, dtype=np.uint8)
        if radius > 1.0:
            stride_x = stride * radius
            stride_y = stride * radius
        else:
            width = gt[2] - gt[0]
            height = gt[3] - gt[1]
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


if __name__ == "__main__":
    cr = CenterReg(288, stride=8, size=23, center_sampling_radius=4)
    cls, reg = cr([80, 60, 95, 140])
    print("test")