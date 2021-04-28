import math
from utils.opbox import *


class Anchors:
    """
    This class generate anchors.
    """
    def __init__(self, stride, ratios, scales):
        self.stride = stride
        self.ratios = ratios
        self.scales = scales

        self.anchor_num = len(self.scales) * len(self.ratios)

        self.anchors = None
        self.generate_anchors()

    def generate_anchors(self):
        """
        generate anchors based on predefined configuration
        """
        self.anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)
        size = self.stride * self.stride
        count = 0
        for r in self.ratios:
            ws = int(math.sqrt(size * 1. / r))
            hs = int(ws * r)
            for s in self.scales:
                w = ws * s
                h = hs * s
                self.anchors[count][:] = [-w * 0.5, -h * 0.5, w * 0.5, h * 0.5][:]
                count += 1

    def generate_all_anchors(self, im_c, size):
        """
        im_c: image center 127
        size: image size 25
        """
        a0x = im_c - size // 2 * self.stride
        ori = np.array([a0x] * 4, dtype=np.float32)
        zero_anchors = self.anchors + ori

        x1 = zero_anchors[:, 0]
        y1 = zero_anchors[:, 1]
        x2 = zero_anchors[:, 2]
        y2 = zero_anchors[:, 3]

        x1, y1, x2, y2 = map(lambda x: x.reshape(self.anchor_num, 1, 1),
                             [x1, y1, x2, y2])
        cx, cy, w, h = corner2center([x1, y1, x2, y2])

        disp_x = np.arange(0, size).reshape(1, 1, -1) * self.stride
        disp_y = np.arange(0, size).reshape(1, -1, 1) * self.stride

        cx = cx + disp_x
        cy = cy + disp_y

        # broadcast
        zero = np.zeros((self.anchor_num, size, size), dtype=np.float32)
        cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])
        x1, y1, x2, y2 = center2corner([cx, cy, w, h])

        self.all_anchors = (np.stack([x1, y1, x2, y2]).astype(np.float32),
                            np.stack([cx, cy, w, h]).astype(np.float32))
        return True


class AnchorTarget:
    def __init__(self, search_sz=255, output_sz=25, stride=8, ratios=None, scales=None):
        self.search_sz = search_sz
        self.output_sz = output_sz
        self.stride = stride
        self.ratios = ratios
        self.scales = scales
        self.anchors = Anchors(stride, ratios, scales)
        self.anchors.generate_all_anchors(search_sz // 2, output_sz)
        self.pos_num = 16
        self.total_num = 64
        self.thr_high = 0.6
        self.thr_low = 0.3
        self.r_pos = 16
        self.r_neg = 0

    def select(self, position, keep_num=16):
        num = position[0].shape[0]
        if num <= keep_num:
            return position, num
        slt = np.arange(num)
        np.random.shuffle(slt)
        slt = slt[:keep_num]
        return tuple(p[slt] for p in position), keep_num

    def __call__(self, target, neg=False):
        """
        target: x1, y1, x2, y2
        """
        size = self.output_sz
        anchor_num = len(self.ratios) * len(self.scales)

        # -1 ignore 0 negative 1 positive
        cls = -1 * np.ones((anchor_num, size, size), dtype=np.int64)
        delta = np.zeros((4, anchor_num, size, size), dtype=np.float32)
        delta_weight = np.zeros((anchor_num, size, size), dtype=np.float32)

        tcx, tcy, tw, th = corner2center(target)
        if neg:
            cx = size // 2
            cy = size // 2
            cx += int(np.ceil((tcx - self.search_sz // 2) /
                              self.stride + 0.5))
            cy += int(np.ceil((tcy - self.search_sz // 2) /
                              self.stride + 0.5))
            l = max(0, cx - 3)
            r = min(size, cx + 4)
            u = max(0, cy - 3)
            d = min(size, cy + 4)
            cls[:, u:d, l:r] = 0

            neg, neg_num = self.select(np.where(cls == 0), self.total_num - self.pos_num)
            cls[:] = -1
            cls[neg] = 0

            overlap = np.zeros((anchor_num, size, size), dtype=np.float32)
            return cls, delta, delta_weight, overlap

        anchor_box = self.anchors.all_anchors[0]
        anchor_center = self.anchors.all_anchors[1]
        x1, y1, x2, y2 = anchor_box[0], anchor_box[1], \
                         anchor_box[2], anchor_box[3]
        cx, cy, w, h = anchor_center[0], anchor_center[1], \
                       anchor_center[2], anchor_center[3]

        delta[0] = (tcx - cx) / w
        delta[1] = (tcy - cy) / h
        delta[2] = np.log(tw / w)
        delta[3] = np.log(th / h)

        overlap = IoU([x1, y1, x2, y2], target)

        pos = np.where(overlap > self.thr_high)
        neg = np.where(overlap < self.thr_low)

        pos, pos_num = self.select(pos, self.pos_num)
        neg, neg_num = self.select(neg, self.total_num - self.pos_num)

        cls[pos] = 1
        delta_weight[pos] = 1. / (pos_num + 1e-6)

        cls[neg] = 0
        return cls, delta, delta_weight, overlap

    def create_fc_labels(self, box, neg=False):
        # distances along x- and y-axis
        h, w = self.output_sz, self.output_sz
        delta = (box[0] + box[2] - self.search_sz) / 2.0 // self.stride, \
                (box[1] + box[3] - self.search_sz) / 2.0 // self.stride
        x = np.arange(w) - (w // 2 + delta[0])
        y = np.arange(h) - (h // 2 + delta[1])
        x, y = np.meshgrid(x, y)

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       -np.ones_like(x),
                                       np.zeros_like(x))).astype(np.float32)
            return labels

        # create logistic labels
        if neg:
            labels = np.zeros_like(x).astype(np.float32)
            weights = np.ones_like(labels, dtype=np.float32)
            delta = 1000, 1000
        else:
            r_pos = self.r_pos / self.stride
            r_neg = self.r_neg / self.stride
            labels = logistic_labels(x, y, r_pos, r_neg)

            # pos/neg weights
            pos_num = np.sum(labels == 1)
            neg_num = np.sum(labels == 0)
            weights = np.zeros_like(labels, dtype=np.float32)
            weights[labels == 1] = 0.5 / pos_num
            weights[labels == 0] = 0.5 / neg_num
            weights *= pos_num + neg_num

        # repeat to size
        labels = labels.reshape((1, h, w))
        weights = weights.reshape((1, h, w))

        return labels, weights, np.array(delta, dtype=np.float32),


# test
if __name__ == "__main__":
    anchor = AnchorTarget(search_sz=255, output_sz=25, stride=8,
                              ratios=[0.33, 0.5, 1, 2, 3], scales=[8])
    box = [20, 40, 80, 100]
    out_sz = 25
    rpn_anchors = anchor(box, neg=True)
    fc_anchors = anchor.create_fc_labels(box, False)
    print("test done!")
