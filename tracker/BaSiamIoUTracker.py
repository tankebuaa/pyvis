import numpy as np
import torch
from torchvision.ops import nms
import torch.nn.functional as F
import cv2
from .base_tracker import BaseTracker
from config import basi_cfg as cfg
from utils.visdom import Visdom
from utils.preprocessing import numpy_to_torch, preprocrss
import time


class BaSiamIoUTracker(BaseTracker):
    def __init__(self, model):
        super(BaSiamIoUTracker, self).__init__()
        hanning = np.hanning(cfg.SCORE_SIZE)
        hann_window = torch.from_numpy(np.outer(hanning, hanning).astype(np.float32)).cuda()
        self.window = hann_window / hann_window.sum()
        step = cfg.STEP
        self.scale_factors = torch.Tensor([[1.0, 1.0, 1.0 - 2 * step, 1.0 - 2 * step],
                                           [1.0, 1.0, 1.0 - step, 1.0 - step],
                                           [1.0, 1.0, 1.0, 1.0],
                                           [1.0, 1.0, 1.0 + step, 1.0 + step],
                                           [1.0, 1.0, 1.0 + 2 * step, 1.0 + 2 * step],]).cuda().float()

        self.model = model
        self.model.eval()
        offset = (cfg.INSTANCE_SIZE - 1) / 2 - cfg.STRIDE * (cfg.SCORE_SIZE // 2)
        self.locations = self.generate_points(cfg.STRIDE, cfg.SCORE_SIZE, offset).cuda()

        self.img_norm_cfg = dict(mean=torch.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).cuda(),
                                 std=torch.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).cuda())

        # debug
        self.debug = cfg.DEBUG
        self.pause_mode = False
        self.step = False
        try:
            self.visdom = Visdom(self.debug, {'handler': self._visdom_ui_handler, 'win_id': 'Tracking'},
                                 visdom_info={'server': '127.0.0.1', 'port': 8097})

            # Show help
            help_text = 'You can pause/unpause the tracker by pressing ''space'' with the ''Tracking'' window ' \
                        'selected. During paused mode, you can track for one frame by pressing the right arrow key.' \
                        'To enable/disable plotting of a data block, tick/untick the corresponding entry in ' \
                        'block list.'
            self.visdom.register(help_text, 'text', 1, 'Help')
        except:
            time.sleep(0.5)
            print('!!! WARNING: Visdom could not start, so using matplotlib visualization instead !!!\n'
                  '!!! Start Visdom in a separate terminal window by typing \'visdom\' !!!')
            self.visdom = None

    def _visdom_ui_handler(self, data):
        if data['event_type'] == 'KeyPress':
            if data['key'] == ' ':
                self.pause_mode = not self.pause_mode

            elif data['key'] == 'ArrowRight' and self.pause_mode:
                self.step = True

    def pause_point(self, delay=0.1):
        while True:
            if not self.pause_mode:
                break
            elif self.step:
                self.step = False
                break
            else:
                time.sleep(delay)

    def generate_points(self, stride, size, offset):
        shifts_x = torch.arange(0, size * stride, step=stride, dtype=torch.float32)
        shifts_y = torch.arange(0, size * stride, step=stride, dtype=torch.float32)
        y, x = torch.meshgrid(shifts_x, shifts_y)
        shift_x = x.reshape(-1)
        shift_y = y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + offset
        return locations

    def get_subwindow(self, image, pos, model_sz, original_sz):
        """
        args:
            image: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
        """
        im = numpy_to_torch(image).cuda()
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

        sz = sample_sz.float() / df     # new size

        # Do downsampling
        if df > 1:
            os = posl % df  # offset
            posl = (posl - os) / df  # new position
            im2 = im[..., os[0].item()::df, os[1].item()::df]  # downsample
        else:
            im2 = im

        # compute size to crop
        szl = torch.max(sz.round(), torch.Tensor([2])).long()

        # Extract top and bottom coordinates
        tl = posl - (szl - 1) / 2
        br = posl + szl / 2 + 1

        im_patch = F.pad(im2,
                         (-tl[1].item(), br[1].item() - im2.shape[3], -tl[0].item(), br[0].item() - im2.shape[2]),
                         'replicate')
        # Resample
        im_patch = F.interpolate(im_patch, output_sz.long().tolist(), mode='bilinear')
        return preprocrss(im_patch, **self.img_norm_cfg)

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.image_sz = np.array([img.shape[1], img.shape[0]], dtype=np.float32)

        # target pos and size
        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                                    bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]], dtype=np.float32)

        # Setup scale change bounds
        self.min_scale_factor = np.max(10 / self.size)
        self.max_scale_factor = np.min(self.image_sz / self.size)

        # template region
        s_z = np.sqrt(self.size.prod()) * cfg.exemplar_area_factor

        # scale from origin to input
        scale = s_z / cfg.EXEMPLAR_SIZE

        # crop template region and resize to EXEMPLAR_SIZE
        # z_crop = self.get_subwindow(img, self.center_pos,
        #                             cfg.EXEMPLAR_SIZE,
        #                             s_z, self.channel_average).cuda()

        s_x = cfg.INSTANCE_SIZE * scale
        x_crop = self.get_subwindow(img, self.center_pos, cfg.INSTANCE_SIZE, s_x)

        z_crop = x_crop.clone()[:, :, cfg.EXEMPLAR_SIZE // 2:cfg.EXEMPLAR_SIZE * 3 // 2,
                 cfg.EXEMPLAR_SIZE // 2:cfg.EXEMPLAR_SIZE * 3 // 2]

        z_bbox = torch.from_numpy(np.array([[cfg.EXEMPLAR_SIZE // 2 - bbox[2] / scale / 2,
                                             cfg.EXEMPLAR_SIZE // 2 - bbox[3] / scale / 2,
                                             bbox[2] / scale, bbox[3] / scale]], dtype=np.float32)).cuda()

        self.model.template(z_crop, z_bbox, x_crop)

        # if self.visdom is not None:
        #     search_area_bbox = [self.center_pos[0] - s_x / 4, self.center_pos[1] - s_x / 4, s_x/2, s_x/2]
        #     self.visdom.register((img[:, :, ::-1], search_area_bbox, list(bbox)), 'Tracking', 1, 'Visual')

    def norm(self, x):
        x = x - x.min()
        x /= x.sum() + 1e-16
        return x

    def track(self, img):
        """
            bbox(list):[x, y, width, height]
        """
        if self.debug and self.visdom:
            self.pause_point()

        """inference"""
        # template size
        self.size_ct = torch.from_numpy(self.size).cuda().float()
        s_z = np.sqrt(self.size.prod()) * cfg.exemplar_area_factor
        self.scale_z = s_z / cfg.EXEMPLAR_SIZE

        # search size
        s_x = self.scale_z * cfg.INSTANCE_SIZE

        # extract search size and resize to INSTANCE_SIZE
        x_crop = self.get_subwindow(img, self.center_pos, cfg.INSTANCE_SIZE, s_x)

        # get classification and regression
        dscores, cls_co, reg = self.model.ba_track(x_crop, self.visdom)

        """tracking"""
        # regressed boxes
        cxcy = self.locations
        ltrb = reg.squeeze().permute(1, 2, 0).reshape(-1, 4)
        boxes = torch.stack((cxcy[:, 0] - ltrb[:, 0], cxcy[:, 1] - ltrb[:, 1],
                             cxcy[:, 0] + ltrb[:, 2], cxcy[:, 1] + ltrb[:, 3]), dim=1)
        iou_boxes = boxes.clone()
        iou_boxes[:, 2:] = iou_boxes[:, 2:] - iou_boxes[:, :2]

        # original cls score
        cls = torch.sum(dscores, dim=0)
        score = self.norm(cls)
        score_co = self.norm(cls_co)

        # box penalty
        penalty = self.cal_penalty(reg, cfg.PENALTY_K)

        # adding cosine window
        score_cls = (1 - cfg.CO_WEIGHT) * score + cfg.CO_WEIGHT * score_co
        scores = (cfg.WINDOW_INFLUENCE * self.window + (1 - cfg.WINDOW_INFLUENCE) * self.norm(score_cls * penalty)) * 100.0

        # draw candidates by nms
        ind1 = torch.where(score_cls.view(-1) > (0.8 * torch.mean(score_cls) + torch.max(score_cls)) / 2)[0]
        ind2 = nms(boxes[ind1], scores.view(-1)[ind1], cfg.NMS_TH)
        candidates = boxes[ind1][ind2]

        # visual debug
        if self.visdom is not None:
            cans = candidates.cpu().numpy()
            search_area_bbox = [self.center_pos[0] - s_x / 2, self.center_pos[1] - s_x / 2, s_x, s_x]
            input = [img[:, :, ::-1], search_area_bbox]
            for i in range(candidates.shape[0]):
                input.append([cans[i, 0] * self.scale_z + search_area_bbox[0],
                              cans[i, 1] * self.scale_z + search_area_bbox[1],
                              (cans[i, 2] - cans[i, 0]) * self.scale_z,
                              (cans[i, 3] - cans[i, 1]) * self.scale_z])
            self.visdom.register(tuple(input), 'Tracking', 2, 'Tracking')
            self.visdom.register(score, 'heatmap', 3, 'raw_map')
            self.visdom.register(score_co, 'heatmap', 3, 'co_map')
            self.visdom.register(score_cls, 'heatmap', 3, 'cls_map')
            self.visdom.register(penalty, 'heatmap', 3, 'penalty')
            self.visdom.register(scores, 'heatmap', 3, 'final_score')

        # final target box
        peak, inds = torch.topk(scores.view(-1)[ind1][ind2], 1)

        # refine by iounet
        if cfg.REFINE:
            # join last bounding box?
            if cfg.BASE_BOX:
                wh = torch.from_numpy(self.size / self.scale_z).cuda().float()
                rbox = torch.cat((cxcy[ind1][ind2][inds][0] - wh / 2, wh), dim=0).unsqueeze(0) * self.scale_factors
                init_boxes = torch.cat((iou_boxes[ind1][ind2][inds], rbox), dim=0)
            else:
                init_boxes = iou_boxes[ind1][ind2][inds] * self.scale_factors
            out_boxes, out_ious = self.model.refine(init_boxes)  # Tensor([x1,y1,w,h])
            iou_score, ind_iou = torch.topk(out_ious, cfg.TOPK)
            predicted_box = out_boxes[ind_iou, :].mean(0).cpu().numpy()
            # visual debug
            if self.visdom is not None:
                outboxes = out_boxes.cpu().numpy()
                search_area_bbox = [self.center_pos[0] - s_x / 2, self.center_pos[1] - s_x / 2, s_x, s_x]
                input = [img[:, :, ::-1], search_area_bbox]
                for i in range(outboxes.shape[0]):
                    input.append([outboxes[i, 0] * self.scale_z + search_area_bbox[0],
                                  outboxes[i, 1] * self.scale_z + search_area_bbox[1],
                                  outboxes[i, 2] * self.scale_z,
                                  outboxes[i, 3] * self.scale_z])
                self.visdom.register(tuple(input), 'Tracking', 3, 'Refine')
        else:
            predicted_box = iou_boxes[ind1][ind2][inds].mean(0).cpu().numpy()

        # detect salient and update vector
        if dscores is not None and candidates.size(0) > 1:
            distractors = ind1[ind2][torch.arange(candidates.size(0)).cuda() != inds]
            distractors = distractors[cls.view(-1)[distractors] > 0.9 * cls.view(-1)[ind1][ind2][inds]]
            if distractors.size(0) > 0:
                centers = torch.stack([distractors % cfg.SCORE_SIZE, distractors / cfg.SCORE_SIZE]).unsqueeze(-1)
                dscore = dscores.view(dscores.shape[0], -1)
                _, maxi = torch.max(dscore, dim=1)
                loc = torch.stack([maxi % cfg.SCORE_SIZE, maxi / cfg.SCORE_SIZE])
                bias = torch.sum(abs(loc.unsqueeze(1).repeat(1, distractors.size(0), 1) - centers), dim=0) <= 2
                self.model.update_salient(bias)
        else:
            self.model.update_salient(None)

        """post processing"""
        # remap bounding-box
        delta = predicted_box[:2] - (cfg.INSTANCE_SIZE - 1) / 2
        xy = delta * self.scale_z + self.center_pos
        wh = cfg.SCALE_LR * predicted_box[2:] * self.scale_z + (1 - cfg.SCALE_LR) * self.size
        cx, cy = xy + wh / 2.0
        wd, ht = wh

        # detect failure
        # if cfg.CHECK_FAILURE and cls.max() < 1000:
            # cx, cy = self.center_pos
            # wd, ht = self.size * 1.05 #predicted_box[2:] * 1.1
        # print(cls.view(-1)[ind1][ind2][inds])

        # clip boundary
        cx, cy, wd, ht = self._bbox_clip(cx, cy, wd, ht, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([wd, ht])

        bbox = [cx - wd / 2,
                cy - ht / 2,
                wd,
                ht]

        if self.visdom is not None:
            self.visdom.register((img[:, :, ::-1], search_area_bbox, bbox), 'Tracking', 1, 'Visual')

        return {
            'bbox': bbox,
            'best_score': 0
        }

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]-5))
        cy = max(0, min(cy, boundary[0]-5))
        width = max(10.0, min(width, boundary[1]-1))
        height = max(10.0, min(height, boundary[0]-1))
        return cx, cy, width, height

    def cal_penalty(self, ltrbs, penalty_lk):
        bboxes_w = ltrbs[0, :, :] + ltrbs[2, :, :]
        bboxes_h = ltrbs[1, :, :] + ltrbs[3, :, :]
        w_c = self.change(ltrbs[0, :, :] / ltrbs[2, :, :])
        h_c = self.change(ltrbs[1, :, :] / ltrbs[3, :, :])
        s_c = self.change(self.sz(bboxes_w, bboxes_h) / self.sz(self.size_ct[0], self.size_ct[1]) / self.scale_z)
        r_c = self.change((self.size_ct[0] / self.size_ct[1]) / (bboxes_w / bboxes_h))
        penalty = torch.exp(-(r_c * s_c - 1) * penalty_lk)#* w_c * h_c + torch.log(w_c * h_c)
        return penalty

    def sz(self, w, h):
        return torch.sqrt(w * h) * cfg.exemplar_area_factor

    def change(self, r):
        return torch.max(r, 1. / r)
