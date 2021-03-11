import visdom
import visdom.server
import cv2
import torch
import copy
import numpy as np
from collections import OrderedDict


def numpy_to_torch(a: np.ndarray):
    return torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0)


def show_image_with_boxes(im, boxes, iou_pred=None, disp_ids=None):
    im_np = im.clone().cpu().squeeze().numpy()
    im_np = np.ascontiguousarray(im_np.transpose(1, 2, 0).astype(np.uint8))

    boxes = boxes.view(-1, 4).cpu().numpy().round().astype(int)

    # Draw proposals
    for i_ in range(boxes.shape[0]):
        if disp_ids is None or disp_ids[i_]:
            bb = boxes[i_, :]
            disp_color = (i_*38 % 256, (255 - i_*97) % 256, (123 + i_*66) % 256)
            cv2.rectangle(im_np, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]),
                          disp_color, 1)

            if iou_pred is not None:
                text_pos = (bb[0], bb[1] - 5)
                cv2.putText(im_np, 'ID={} IOU = {:3.2f}'.format(i_, iou_pred[i_]), text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, bottomLeftOrigin=False)

    im_tensor = torch.from_numpy(im_np.transpose(2, 0, 1)).float()

    return im_tensor


def _pascal_color_map(N=256, normalized=False):
    """
    Python implementation of the color map function for the PASCAL VOC data set.
    Official Matlab version can be found in the PASCAL VOC devkit
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def overlay_mask(im, ann, alpha=0.5, colors=None, contour_thickness=None):
    """ Overlay mask over image.
    Source: https://github.com/albertomontesg/davis-interactive/blob/master/davisinteractive/utils/visualization.py
    This function allows you to overlay a mask over an image with some
    transparency.
    # Arguments
        im: Numpy Array. Array with the image. The shape must be (H, W, 3) and
            the pixels must be represented as `np.uint8` data type.
        ann: Numpy Array. Array with the mask. The shape must be (H, W) and the
            values must be intergers
        alpha: Float. Proportion of alpha to apply at the overlaid mask.
        colors: Numpy Array. Optional custom colormap. It must have shape (N, 3)
            being N the maximum number of colors to represent.
        contour_thickness: Integer. Thickness of each object index contour draw
            over the overlay. This function requires to have installed the
            package `opencv-python`.
    # Returns
        Numpy Array: Image of the overlay with shape (H, W, 3) and data type
            `np.uint8`.
    """
    im, ann = np.asarray(im, dtype=np.uint8), np.asarray(ann, dtype=np.int)
    if im.shape[:-1] != ann.shape:
        raise ValueError('First two dimensions of `im` and `ann` must match')
    if im.shape[-1] != 3:
        raise ValueError('im must have three channels at the 3 dimension')

    colors = colors or _pascal_color_map()
    colors = np.asarray(colors, dtype=np.uint8)

    mask = colors[ann]
    fg = im * alpha + (1 - alpha) * mask

    img = im.copy()
    img[ann > 0] = fg[ann > 0]

    if contour_thickness:  # pragma: no cover
        import cv2
        for obj_id in np.unique(ann[ann > 0]):
            contours = cv2.findContours((ann == obj_id).astype(
                np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            cv2.drawContours(img, contours[0], -1, colors[obj_id].tolist(),
                             contour_thickness)
    return img


class VisBase:
    def __init__(self, visdom, show_data, title):
        self.visdom = visdom
        self.show_data = show_data
        self.title = title
        self.raw_data = None

    def update(self, data, **kwargs):
        self.save_data(data, **kwargs)

        if self.show_data:
            self.draw_data()

    def save_data(self, data, **kwargs):
        raise NotImplementedError

    def draw_data(self):
        raise NotImplementedError

    def toggle_display(self, new_mode=None):
        if new_mode is not None:
            self.show_data = new_mode
        else:
            self.show_data = not self.show_data

        if self.show_data:
            self.draw_data()
        else:
            self.visdom.close(self.title)


class VisImage(VisBase):
    def __init__(self, visdom, show_data, title):
        super().__init__(visdom, show_data, title)

    def save_data(self, data):
        data = data.float()
        self.raw_data = data

    def draw_data(self):
        self.visdom.image(self.raw_data.clone(), opts={'title': self.title}, win=self.title)


class VisHeatmap(VisBase):
    def __init__(self, visdom, show_data, title):
        super().__init__(visdom, show_data, title)

    def save_data(self, data):
        data = data.squeeze().flip(0)
        self.raw_data = data

    def draw_data(self):
        self.visdom.heatmap(self.raw_data.clone(),  opts={'title': self.title}, win=self.title)


class VisFeaturemap(VisBase):
    def __init__(self, visdom, show_data, title):
        super().__init__(visdom, show_data, title)
        self.block_list = None

    def block_list_callback_handler(self, data):
        self.block_list[data['propertyId']]['value'] = data['value']
        self.visdom.properties(self.block_list, opts={'title': 'Featuremap UI'}, win='featuremap_ui')
        self.draw_data()

    def save_data(self, data):
        data = data.view(-1, *data.shape[-2:])
        data = data.flip(1)
        if self.block_list is None:
            self.block_list = []
            self.draw_feat = []
            for i in range(data.shape[0]):
                self.block_list.append({'type': 'checkbox', 'name': 'Channel {:04d}'.format(i), 'value': False})

            self.visdom.properties(self.block_list, opts={'title': 'Featuremap UI'}, win='featuremap_ui')
            self.visdom.register_event_handler(self.block_list_callback_handler, 'featuremap_ui')

        self.raw_data = data

    def draw_data(self):
        if self.block_list is not None and self.show_data:
            for i, d in enumerate(self.block_list):
                if d['value']:
                    fig_title = '{} ch: {:04d}'.format(self.title, i)
                    self.visdom.heatmap(self.raw_data[i, :, :].clone(),
                                        opts={'title': fig_title}, win=fig_title)


class VisCostVolume(VisBase):
    def __init__(self, visdom, show_data, title, flip=False):
        super().__init__(visdom, show_data, title)
        self.show_slice = False
        self.slice_pos = None
        self.flip = flip

    def show_cost_volume(self):
        data = self.raw_data.clone()

        # data_perm = data.permute(2, 0, 3, 1).contiguous()
        data_perm = data.permute(0, 2, 1, 3).contiguous()
        if self.flip:
            data_perm = data_perm.permute(2, 3, 0, 1).contiguous()

        data_perm = data_perm.view(data_perm.shape[0] * data_perm.shape[1], -1)
        self.visdom.heatmap(data_perm.flip(0), opts={'title': self.title}, win=self.title)

    def set_zoom_pos(self, slice_pos):
        self.slice_pos = slice_pos

    def toggle_show_slice(self, new_mode=None):
        if new_mode is not None:
            self.show_slice = new_mode
        else:
            self.show_slice = not self.show_slice

    def show_cost_volume_slice(self):
        slice_pos = self.slice_pos

        # slice_pos: [row, col]
        cost_volume_data = self.raw_data.clone()

        if self.flip:
            cost_volume_slice = cost_volume_data[:, :, slice_pos[0], slice_pos[1]]
        else:
            cost_volume_slice = cost_volume_data[slice_pos[0], slice_pos[1], :, :]
        self.visdom.heatmap(cost_volume_slice.flip(0), opts={'title': self.title}, win=self.title)

    def save_data(self, data):
        data = data.view(data.shape[-2], data.shape[-1], data.shape[-2], data.shape[-1])
        self.raw_data = data

    def draw_data(self):
        if self.show_slice:
            self.show_cost_volume_slice()
        else:
            self.show_cost_volume()


class VisCostVolumeUI(VisBase):
    def cv_ui_handler(self, data):
        zoom_toggled = False
        if data['event_type'] == 'KeyPress':
            if data['key'] == 'ArrowRight':
                self.zoom_pos[1] = min(self.zoom_pos[1] + 1, self.feat_shape[1]-1)
            elif data['key'] == 'ArrowLeft':
                self.zoom_pos[1] = max(self.zoom_pos[1] - 1, 0)
            elif data['key'] == 'ArrowUp':
                self.zoom_pos[0] = max(self.zoom_pos[0] - 1, 0)
            elif data['key'] == 'ArrowDown':
                self.zoom_pos[0] = min(self.zoom_pos[0] + 1, self.feat_shape[0]-1)
            elif data['key'] == 'Enter':
                self.zoom_mode = not self.zoom_mode
                zoom_toggled = True

        # Update image
        self.show_image()

        # Update cost volumes
        for block_title, block in self.registered_blocks.items():
            if isinstance(block, VisCostVolume):
                block.set_zoom_pos(self.zoom_pos)
                block.toggle_show_slice(self.zoom_mode)

                if (self.zoom_mode or zoom_toggled) and block.show_data:
                    block.draw_data()

    def __init__(self, visdom, show_data, title, feat_shape, registered_blocks):
        super().__init__(visdom, show_data, title)
        self.feat_shape = feat_shape
        self.zoom_mode = False
        self.zoom_pos = [int((feat_shape[0] - 1) / 2), int((feat_shape[1] - 1) / 2)]
        self.registered_blocks = registered_blocks

        self.visdom.register_event_handler(self.cv_ui_handler, title)

    def draw_grid(self, data):
        stride_r = int(data.shape[1] / self.feat_shape[0])
        stride_c = int(data.shape[2] / self.feat_shape[1])

        # Draw grid
        data[:, list(range(0, data.shape[1], stride_r)), :] = 0
        data[:, :, list(range(0, data.shape[2], stride_c))] = 0

        data[0, list(range(0, data.shape[1], stride_r)), :] = 255
        data[0, :, list(range(0, data.shape[2], stride_c))] = 255

        return data

    def shade_cell(self, data):
        stride_r = int(data.shape[1] / self.feat_shape[0])
        stride_c = int(data.shape[2] / self.feat_shape[1])

        r1 = self.zoom_pos[0]*stride_r
        r2 = min((self.zoom_pos[0] + 1)*stride_r, data.shape[1])

        c1 = self.zoom_pos[1] * stride_c
        c2 = min((self.zoom_pos[1] + 1) * stride_c, data.shape[2])

        factor = 0.8 if self.zoom_mode else 0.5
        data[:, r1:r2, c1:c2] = data[:, r1:r2, c1:c2] * (1 - factor) + torch.tensor([255.0, 0.0, 0.0]).view(3, 1, 1).to(data.device) * factor
        return data

    def show_image(self, data=None):
        if data is None:
            data = self.raw_data.clone()

        data = self.draw_grid(data)
        data = self.shade_cell(data)
        self.visdom.image(data, opts={'title': self.title}, win=self.title)

    def save_data(self, data):
        # Ignore feat shape
        data = data[0]
        data = data.float()
        self.raw_data = data

    def draw_data(self):
        self.show_image(self.raw_data.clone())


class VisInfoDict(VisBase):
    def __init__(self, visdom, show_data, title):
        super().__init__(visdom, show_data, title)
        self.raw_data = OrderedDict()

    def generate_display_text(self, data):
        display_text = ''
        for key, value in data.items():
            key = key.replace('_', ' ')
            if value is None:
                display_text += '<b>{}</b>: {}<br>'.format(key, 'None')
            elif isinstance(value, (str, int)):
                display_text += '<b>{}</b>: {}<br>'.format(key, value)
            else:
                display_text += '<b>{}</b>: {:.2f}<br>'.format(key, value)

        return display_text

    def save_data(self, data):
        for key, val in data.items():
            self.raw_data[key] = val

    def draw_data(self):
        data = copy.deepcopy(self.raw_data)
        display_text = self.generate_display_text(data)
        self.visdom.text(display_text, opts={'title': self.title}, win=self.title)


class VisText(VisBase):
    def __init__(self, visdom, show_data, title):
        super().__init__(visdom, show_data, title)

    def save_data(self, data):
        self.raw_data = data

    def draw_data(self):
        data = copy.deepcopy(self.raw_data)
        self.visdom.text(data, opts={'title': self.title}, win=self.title)


class VisLinePlot(VisBase):
    def __init__(self, visdom, show_data, title):
        super().__init__(visdom, show_data, title)

    def save_data(self, data):
        self.raw_data = data

    def draw_data(self):
        if isinstance(self.raw_data, (list, tuple)):
            data_y = self.raw_data[0].clone()
            data_x = self.raw_data[1].clone()
        else:
            data_y = self.raw_data.clone()
            data_x = torch.arange(data_y.shape[0])

        self.visdom.line(data_y, data_x, opts={'title': self.title}, win=self.title)


class VisTracking(VisBase):
    def __init__(self, visdom, show_data, title):
        super().__init__(visdom, show_data, title)

    def save_data(self, data):
        image = data[0]
        boxes_masks = data[1:]

        boxes, masks = [], []
        for bm in boxes_masks:
            if bm is None:
                continue
            if isinstance(bm, list):
                boxes.append(torch.Tensor(bm)); continue
            if len(bm.shape) > 1:
                # Binarize segmentation if a float tensor is provided
                if bm.dtype != np.uint8:
                    bm = (bm > 0.5).astype(np.uint8)
                masks.append(bm); continue
            boxes.append(bm.float())

        self.raw_data = [image, boxes, masks]

    def draw_data(self):
        disp_image = self.raw_data[0].copy()

        resize_factor = 1
        if max(disp_image.shape) > 480:
            resize_factor = 480.0 / float(max(disp_image.shape))
            disp_image = cv2.resize(disp_image, None, fx=resize_factor, fy=resize_factor)
            for i, mask in enumerate(self.raw_data[2]):
                self.raw_data[2][i] = cv2.resize(mask, None, fx=resize_factor, fy=resize_factor)

        boxes = [resize_factor * b.clone() for b in self.raw_data[1]]

        for i, disp_rect in enumerate(boxes):#0-black;1r;2b;
            color = ((255*(i%2)), 255*((i%4)//2), 255*((i%8)//4))
            cv2.rectangle(disp_image,
                          (int(disp_rect[0]), int(disp_rect[1])),
                          (int(disp_rect[0] + disp_rect[2]), int(disp_rect[1] + disp_rect[3])), color, 2)
        for i, mask in enumerate(self.raw_data[2], 1):
            disp_image = overlay_mask(disp_image, mask * i)
        disp_image = numpy_to_torch(disp_image).squeeze(0)
        disp_image = disp_image.float()
        self.visdom.image(disp_image, opts={'title': self.title}, win=self.title)


class VisBBReg(VisBase):
    def __init__(self, visdom, show_data, title):
        super().__init__(visdom, show_data, title)
        self.block_list = []

    def block_list_callback_handler(self, data):
        self.block_list[data['propertyId']]['value'] = data['value']
        self.visdom.properties(self.block_list, opts={'title': 'BBReg Vis'}, win='bbreg_vis')
        self.draw_data()

    def save_data(self, data):
        self.image = data[0].float()
        self.init_boxes = data[1]
        self.final_boxes = data[2]
        self.final_ious = data[3]

    def draw_data(self):
        if len(self.block_list) == 0:
            self.block_list.append({'type': 'checkbox', 'name': 'ID 0', 'value': True})
            self.block_list.append({'type': 'checkbox', 'name': 'ID 1', 'value': True})
            self.visdom.properties(self.block_list, opts={'title': 'BBReg Vis'}, win='bbreg_vis')
            self.visdom.register_event_handler(self.block_list_callback_handler, 'bbreg_vis')

        disp_image = self.image

        ids = [x['value'] for x in self.block_list]
        init_box_image = show_image_with_boxes(disp_image.clone(), self.init_boxes.clone(), disp_ids=ids)
        final_box_image = show_image_with_boxes(disp_image.clone(), self.final_boxes.clone(), self.final_ious.clone(), disp_ids=ids)

        self.visdom.image(init_box_image, opts={'title': 'Init Boxes'}, win='Init Boxes')
        self.visdom.image(final_box_image, opts={'title': 'Final Boxes'}, win='Final Boxes')


class Visdom:
    def __init__(self, debug=0, ui_info=None, visdom_info=None):
        self.debug = debug
        self.visdom = visdom.Visdom(server=visdom_info.get('server', '127.0.0.1'), port=visdom_info.get('port', 8097))
        self.registered_blocks = {}
        self.blocks_list = []

        self.visdom.properties(self.blocks_list, opts={'title': 'Block List'}, win='block_list')
        self.visdom.register_event_handler(self.block_list_callback_handler, 'block_list')

        if ui_info is not None:
            self.visdom.register_event_handler(ui_info['handler'], ui_info['win_id'])

    def block_list_callback_handler(self, data):
        field_name = self.blocks_list[data['propertyId']]['name']

        self.registered_blocks[field_name].toggle_display(data['value'])

        self.blocks_list[data['propertyId']]['value'] = data['value']

        self.visdom.properties(self.blocks_list, opts={'title': 'Block List'}, win='block_list')

    def register(self, data, mode, debug_level=0, title='Data', **kwargs):
        if title not in self.registered_blocks.keys():
            show_data = self.debug >= debug_level

            if title != 'Tracking':
                self.blocks_list.append({'type': 'checkbox', 'name': title, 'value': show_data})

            self.visdom.properties(self.blocks_list, opts={'title': 'Block List'}, win='block_list')

            if mode == 'image':
                self.registered_blocks[title] = VisImage(self.visdom, show_data, title)
            elif mode == 'heatmap':
                self.registered_blocks[title] = VisHeatmap(self.visdom, show_data, title)
            elif mode == 'cost_volume':
                self.registered_blocks[title] = VisCostVolume(self.visdom, show_data, title)
            elif mode == 'cost_volume_flip':
                self.registered_blocks[title] = VisCostVolume(self.visdom, show_data, title, flip=True)
            elif mode == 'cost_volume_ui':
                self.registered_blocks[title] = VisCostVolumeUI(self.visdom, show_data, title, data[1],
                                                                self.registered_blocks)
            elif mode == 'info_dict':
                self.registered_blocks[title] = VisInfoDict(self.visdom, show_data, title)
            elif mode == 'text':
                self.registered_blocks[title] = VisText(self.visdom, show_data, title)
            elif mode == 'lineplot':
                self.registered_blocks[title] = VisLinePlot(self.visdom, show_data, title)
            elif mode == 'Tracking':
                self.registered_blocks[title] = VisTracking(self.visdom, show_data, title)
            elif mode == 'bbreg':
                self.registered_blocks[title] = VisBBReg(self.visdom, show_data, title)
            elif mode == 'featmap':
                self.registered_blocks[title] = VisFeaturemap(self.visdom, show_data, title)
            else:
                raise ValueError('Visdom Error: Unknown data mode {}'.format(mode))
        # Update
        self.registered_blocks[title].update(data, **kwargs)

