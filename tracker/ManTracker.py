import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tracker.base_tracker import BaseTracker
from config import man_cfg as cfg
import matplotlib.pyplot as plt
import matplotlib.colors as colors


class ManTracker(BaseTracker):
    def __init__(self, model):
        super(ManTracker, self).__init__()
        self.upscale_sz = cfg.SCORE_SIZE * cfg.SCORE_UP
        hanning = np.hanning(self.upscale_sz)
        self.hann_window = np.outer(hanning, hanning)
        self.hann_window /= self.hann_window.sum()
        # search scale factors
        self.scale_factors = cfg.SCALE_STEP **\
                             np.linspace(-(cfg.SCALE_MUN // 2), cfg.SCALE_MUN // 2, cfg.SCALE_MUN)
        self.model = model
        self.model.eval()
        if cfg.mam_visual:
            self.fig, self.axes = plt.subplots(2,3, dpi=72, figsize=(8, 4))
            self.visual_ready = False

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2],dtype=float)
        self.size = np.array([bbox[2], bbox[3]], dtype=float)
        # calculate z crop size
        w_z = self.size[0] + cfg.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
        # get crop
        z_crop = self.get_multi_subwindow(img, self.center_pos, cfg.EXEMPLAR_SIZE, s_z)[0].cuda()
        self.model.template(z_crop)

        # visualization
        self.template_im = z_crop.squeeze().permute(1,2,0).cpu().numpy() / 255.0

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

        s_x = s_z * (cfg.INSTANCE_SIZE / cfg.EXEMPLAR_SIZE)
        x_crops = self.get_multi_subwindow(img, self.center_pos, cfg.INSTANCE_SIZE, s_x, self.scale_factors)
        x_crop = torch.stack(x_crops, dim=0).squeeze().cuda()
        outputs = self.model.track(x_crop)

        ###################MAN Tracking outputs #######################
        responses = outputs['response'].squeeze().cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            t, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC) for t in responses], axis=0)

        responses[:cfg.SCALE_MUN // 2] *= cfg.SCALE_PENALTY
        responses[cfg.SCALE_MUN // 2 + 1:] *= cfg.SCALE_PENALTY

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - cfg.WINDOW_INFLUENCE) * response + cfg.WINDOW_INFLUENCE * self.hann_window
        best_score = response.argmax()
        loc = np.unravel_index(best_score, response.shape)

        # locate target center
        disp_in_response = np.array(loc, dtype=float) - self.upscale_sz // 2
        disp_in_instance = disp_in_response * cfg.STRIDE / cfg.SCORE_UP
        disp_in_image = disp_in_instance * s_x * self.scale_factors[scale_id] / cfg.INSTANCE_SIZE
        self.center_pos += disp_in_image[::-1]

        # update target size
        scale = (1 - cfg.SCALE_LR) * 1.0 + cfg.SCALE_LR * self.scale_factors[scale_id]
        self.size *= scale

        cx = self.center_pos[0]
        cy = self.center_pos[1]
        width = self.size[0]
        height = self.size[1]

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy], dtype=float)
        self.size = np.array([width, height], dtype=float)

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        if not cfg.mam_visual:
            return {
                'bbox': bbox,
                'best_score': best_score
            }

        ################################Visualization################################
        im00 = self.template_im[:, :, ::-1]
        x_img = x_crops[1].squeeze().permute(1,2,0).cpu().numpy()[:, :, ::-1]
        im01 = x_img / 255.0
        MAM = self.model.visualize(x_crops[1]).squeeze().cpu().numpy()
        im02 = MAM
        hmx = cv2.applyColorMap(cv2.resize(np.uint8(255 * MAM), (255, 255), interpolation=1),
                                cv2.COLORMAP_JET)[:, :, ::-1]
        mam_fuse = cv2.addWeighted(x_img, 0.4, hmx.astype('float32'), 0.6, 0.0)
        im10 = mam_fuse / 255.0
        Mx = cv2.resize(MAM, (255, 255), interpolation=1)
        im11 = np.flipud(Mx)
        palette = plt.cm.cool
        palette.set_over('b', 1.0)
        if not self.visual_ready:
            self.visual_ready = True
            for ax in self.fig.axes:
                ax.get_xaxis().set_visible(False)
            for ax in self.fig.axes:
                ax.get_yaxis().set_visible(False)
            self.ax00 = self.axes[0, 0].imshow(im00)
            self.ax01 = self.axes[0, 1].imshow(im01)
            self.ax02 = self.axes[0, 2].imshow(im02)
            self.ax10 = self.axes[1, 0].imshow(im10, cmap=plt.cm.viridis)
            self.ax11 = self.axes[1, 1].imshow(im11, interpolation='bilinear', cmap=palette,
                                norm=colors.BoundaryNorm([0.0, 0.5, 0.6, 0.7, 0.8, 0.9], ncolors=palette.N),
                                aspect='auto',
                                origin='lower')
            self.fig.colorbar(self.ax02, ax=self.axes[0, 2])
            self.fig.colorbar(self.ax11, extend='both', shrink=0.9, ax=self.axes[1, 1])
        else:
            self.ax00.set_data(im00)
            self.ax01.set_data(im01)
            self.ax02.set_data(im02)
            self.ax10.set_data(im10)
            self.ax11.set_data(im11)
        self.fig.canvas.draw_idle()
        plt.pause(0.1)

        return {
                'bbox': bbox,
                'best_score': best_score
               }

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def get_multi_subwindow(self, im, pos, model_sz, original_sz, scales=[1.0, ]):
        imt = torch.from_numpy(im).cuda()
        im = imt.float().permute(2, 0, 1).unsqueeze(0).cuda()
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
        else:
            im2 = im

        # compute multi-scaled sizes to crop
        im_patches = []
        for scale in scales:
            szs = sz * scale
            szl = torch.max(szs.round(), torch.Tensor([2])).long()

            # Extract top and bottom coordinates
            tl = posl - (szl - 1) // 2
            br = posl + szl // 2 + 1

            im_patch = F.pad(im2, [-tl[1].item(), br[1].item() - im2.shape[3],
                                   -tl[0].item(), br[0].item() - im2.shape[2]], mode='replicate')
            # Resample
            im_patch = F.interpolate(im_patch, output_sz.long().tolist(), mode='bilinear')
            im_patches.append(im_patch)

        return im_patches