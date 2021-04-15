import numpy as np
import torch
import torch.nn.functional as F
import cv2
from .base_tracker import BaseTracker
from config import cfg


class IMSiamTracker(BaseTracker):
    def __init__(self, model):
        super(IMSiamTracker, self).__init__()
        hanning = np.hanning(cfg.SCORE_SIZE)
        self.window = np.outer(hanning, hanning)
        self.model = model
        self.model.eval()
        self.locations = self.generate_points(cfg.STRIDE, cfg.SCORE_SIZE, cfg.OFFSET)

    def generate_points(self, stride, size, offset):
        shifts_x = torch.arange(0, size * stride, step=stride, dtype=torch.float32)
        shifts_y = torch.arange(0, size * stride, step=stride, dtype=torch.float32)
        y, x = torch.meshgrid(shifts_x, shifts_y)
        shift_x = x.reshape(-1)
        shift_y = y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + offset
        return locations

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                             int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                          int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        return im_patch

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                                    bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.EXEMPLAR_SIZE,
                                    s_z, self.channel_average).cuda()
        scale = s_z / cfg.EXEMPLAR_SIZE
        z_bbox = torch.from_numpy(np.array([[cfg.EXEMPLAR_SIZE // 2 - bbox[2] / scale / 2,
                                             cfg.EXEMPLAR_SIZE // 2 - bbox[3] / scale / 2,
                                             cfg.EXEMPLAR_SIZE // 2 + bbox[2] / scale / 2,
                                             cfg.EXEMPLAR_SIZE // 2 + bbox[3] / scale / 2]],
                                           dtype=np.float32)).cuda()
        self.model.template(z_crop, z_bbox)

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        self.scale_z = cfg.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.INSTANCE_SIZE / cfg.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.INSTANCE_SIZE,
                                    round(s_x), self.channel_average).cuda()

        cls_s, cen_s, reg_s, iou_s = self.model.track(x_crop)
        score_cls = []
        regs = []
        for cls, cen, reg, iou in zip(cls_s, cen_s, reg_s, iou_s):
            cls = self._convert_cls(cls)
            cen = self._convert_sigmoid(cen)
            reg = reg.squeeze().cpu().numpy()
            iou = self._convert_sigmoid(iou)
            cls = cls  * cen  * iou
            score_cls.append(cls[np.newaxis, :])
            regs.append(reg[np.newaxis, :])
        cls_raw = np.mean(score_cls, axis=0).squeeze()
        cls = cls_raw * (1 - cfg.WINDOW_INFLUENCE) + self.window * cfg.WINDOW_INFLUENCE
        best_score = cls.argmax()
        cr, cl = np.unravel_index(best_score, cls.shape)
        reg = np.concatenate(regs, axis=0)
        ltrb = np.mean(reg[:, :, cr, cl].squeeze(), axis=0) / self.scale_z
        disp = np.array([cl, cr]) - (np.array([cfg.SCORE_SIZE, cfg.SCORE_SIZE]) - 1.) / 2.
        disp_ori = disp * cfg.STRIDE / self.scale_z + self.center_pos
        cbox = np.array([disp_ori[0] - ltrb[0], disp_ori[1] - ltrb[1],
                         disp_ori[0] + ltrb[2], disp_ori[1] + ltrb[3]])
        cx = (cbox[0] + cbox[2]) /2.
        cy = (cbox[1] + cbox[3]) /2.
        wd = cbox[2] - cbox[0]
        ht = cbox[3] - cbox[1]

        # smooth bbox
        wd = self.size[0] * (1 - cfg.SCALE_LR) + wd * cfg.SCALE_LR
        ht = self.size[1] * (1 - cfg.SCALE_LR) + ht * cfg.SCALE_LR

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, wd,
                                                ht, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        return {
            'bbox': bbox,
            'best_score': cls_raw.max()
        }

    def _convert_cls(self, cls):
        cls = F.softmax(cls[:, :, :, :], dim=1).data[:, 1, :, :].squeeze().cpu().numpy()
        return cls

    def _convert_sigmoid(selfself, score):
        score = torch.sigmoid(score).squeeze().cpu().numpy()
        return score

    def sz(self, w, h):
        pad = (w + h) * 0.5
        return np.sqrt((w + pad) * (h + pad))

    def change(self,r):
        return np.maximum(r, 1. / r)

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height