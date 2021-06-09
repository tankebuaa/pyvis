import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
from tracker.base_tracker import BaseTracker
from datasets.anchor import Anchors
from config import cosiamrpn_cfg as cfg


class CoSiamRPNTracker(BaseTracker):
    def __init__(self, model):
        super(CoSiamRPNTracker, self).__init__()
        self.anchor_num = len(cfg.ANCHOR_RATIOS) * len(cfg.ANCHOR_SCALES)
        hanning = np.hanning(cfg.SCORE_SIZE)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.model = model
        self.model.eval()
        self.anchors = self.generate_anchor(cfg.SCORE_SIZE)
        self.img_norm_cfg = dict(mean=torch.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).cuda(),
                                 std=torch.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).cuda())

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
        self.channel_average = None  # np.mean(img, axis=(0, 1))
        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.EXEMPLAR_SIZE,
                                    s_z, self.channel_average).cuda()
        self.model.template(z_crop)

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
        scale_z = cfg.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.INSTANCE_SIZE / cfg.EXEMPLAR_SIZE)

        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.INSTANCE_SIZE,
                                    round(s_x), self.channel_average).cuda()
        outputs = self.model.track(x_crop)

        # MAN outputs
        response = outputs['response'].cpu().numpy()
        response = np.tile(response.flatten(), self.anchor_num)

        # CoRPN outputs
        cls1 = self._convert_score(outputs['cls1'])
        cls2 = self._convert_score(outputs['cls2'])
        cls = cfg.CO_WEIGHT * cls1 + (1 - cfg.CO_WEIGHT) * cls2
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z)))
        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.PENALTY_K)

        # fused cls score
        pcls = penalty * cls

        # final score
        score = (1 - cfg.WINDOW_INFLUENCE) * self._norm_score(pcls) * + \
            cfg.WINDOW_INFLUENCE * self.window + cfg.MAN_INFLUENCE * self._norm_score(response)

        # post processing
        best_idx = np.argmax(score)
        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.SCALE_LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # detect failure
        # if cfg.CHECK_FAILURE and cls2.max() < 0.1:
        #     cx, cy = self.center_pos
        #     width, height = self.size * 1.1 #predicted_box[2:] * 1.1

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = cls[best_idx]

        return {
            'bbox': bbox,
            'best_score': best_score
        }

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.STRIDE,
                          cfg.ANCHOR_RATIOS,
                          cfg.ANCHOR_SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
                 np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _norm_score(self, x):
        x = x - x.min()
        res = x / x.max()
        return res

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def get_subwindow(self, image, pos, model_sz, original_sz, avg_chans=None):
        image = torch.from_numpy(image).cuda()
        im = image.float().permute(2, 0, 1).unsqueeze(0)
        pos = torch.Tensor([pos[1], pos[0]]).round()
        posl = pos.long().clone()

        sample_sz = torch.Tensor([original_sz, original_sz])
        output_sz = torch.Tensor([model_sz, model_sz])

        # Compute pre-downsampling factor
        if output_sz is not None:
            resize_factor = torch.min(sample_sz.float() / output_sz.float()).item()
            df = int(max(int(resize_factor - 0.1), 1))
        else:
            df = int(1)

        sz = sample_sz.float() / df  # new size

        # Do downsampling
        if df > 1:
            os = posl % df  # offset
            posl = (posl - os) // df  # new position
            im2 = im[..., os[0].item()::df, os[1].item()::df]  # downsample
            # im2 = F.interpolate(im, scale_factor=1.0/df, mode='bilinear', align_corners=False)
        else:
            im2 = im

        # compute size to crop
        szl = torch.max(sz.round(), torch.Tensor([2])).long()

        # Extract top and bottom coordinates
        tl = posl - (szl - 1) // 2
        br = posl + szl // 2 + 1

        im_patch = F.pad(im2, [-tl[1].item(), br[1].item() - im2.shape[3],
                               -tl[0].item(), br[0].item() - im2.shape[2]], mode='replicate')  # replicate
        # Resample
        im_patch = F.interpolate(im_patch, output_sz.long().tolist(), mode='bilinear')
        if self.model.backbone_type is 'resnet50c3':
            return im_patch
        else:
            return self.preprocrss(im_patch, **self.img_norm_cfg)

    def preprocrss(self, im: torch.Tensor, mean, std):
        im = im[:, [2, 1, 0], :, :] / 255  # bgr->rgb & normlize
        return (im - mean) / std

    def get_subwindow1(self, im, pos, model_sz, original_sz, avg_chans):
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

        if self.model.backbone_type is 'resnet50c3':
            if not np.array_equal(model_sz, original_sz):
                im_patch = cv2.resize(im_patch, (model_sz, model_sz))
            im_patch = im_patch.astype(np.float32)
            im_patch = im_patch[:, :, ::-1].transpose(2, 0, 1)
            im_patch = im_patch[np.newaxis, :, :, :]
            im_patch = torch.from_numpy(im_patch).cuda()
        else:
            im_patch = torch.from_numpy(im_patch).float().permute(2, 0, 1).unsqueeze(0)
            if not np.array_equal(model_sz, original_sz):
                im_patch = F.interpolate(im_patch, model_sz, mode='bilinear')
            im_patch = self.preprocrss(im_patch, **self.img_norm_cfg)

        return im_patch
