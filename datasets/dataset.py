import numpy as np
import random
import cv2
from torch.utils.data import Dataset
from datasets.anchor import AnchorTarget
from datasets.point_target import PointTarget
from datasets.center_reg import CenterReg
from datasets.loc_label import LocLabel
from utils.opbox import get_min_max_bbox, Center, center2corner, perturb_box
from datasets.augmentation import Augmentation, SSFAugument
import torchvision.transforms as T
from utils.gaussian_blur import GaussianBlur


class SrtDataset(Dataset):
    """
    Draw pair-wise images as training samples for RPN
    """
    def __init__(self, datasets, p_datasets=None, samples_per_epoch=None, train=True, normlization=False):
        super(SrtDataset, self).__init__()
        self.train = train

        self.exemplar_size = 127
        self.search_size = 255
        self.datasets = datasets

        if samples_per_epoch is not None:
            self.samples_per_epoch = samples_per_epoch
        else:
            self.samples_per_epoch = sum([d.num for d in self.datasets])

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]
        self.neg_factor = 0.2 if train else 0.0

        self.template_aumentation = SSFAugument(shift=4, scale=0.05, flip=0.0)
        self.search_aumentation = SSFAugument(shift=64, scale=0.18, flip=0.01 if train else 0.0)#14
        self.transform_train = T.Compose([
            T.ToPILImage(),
            T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.2),
            T.RandomGrayscale(p=0.1),
            T.RandomApply([GaussianBlur(kernel_size=self.search_size // 20 * 2 + 1, sigma=(0.1, 2.0))], p=0.1),
        ])
        self.transform_val = T.Compose([
            T.ToPILImage(),
        ])

        self.norm_out = normlization
        if normlization:
            imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
            self.transorm_norm = T.Compose([
                T.ToTensor(),
                T.Normalize(*imagenet_mean_std)
            ])

        self.anchor_target = AnchorTarget(search_sz=255, output_sz=25,
                                          stride=8, ratios=[0.33, 0.5, 1, 2, 3], scales=[8])

    def __getitem__(self, index):
        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]
        index = random.randint(0, dataset.num - 1)
        neg = self.neg_factor > random.random()

        # get one dataset
        if neg:
            template = dataset.get_random_target()
            search = random.choices(self.datasets, self.p_datasets)[0].get_random_target()
        else:
            template, search = dataset.get_positive_pair(index)

        # get image
        template_image = cv2.imread(template[0])  # bgr
        search_image = cv2.imread(search[0])  # bgr

        # get sample and bounding box (img: bgr, bbox:[x1,y1,x2,y2])
        template, template_bbox, _ = self.template_aumentation(template_image, template[1], self.exemplar_size,
                                                      self.exemplar_size, mode='padding')  # padding
        search, search_bbox, _ = self.search_aumentation(search_image, search[1], self.exemplar_size,
                                                     self.search_size, mode='padding')  # padding
        if self.train:
            template = self.transform_train(template[:, :, ::-1])   # bgr->rgb
            search = self.transform_train(search[:, :, ::-1])   # bgr->rgb
        else:
            template = self.transform_val(template[:, :, ::-1])  # bgr->rgb
            search = self.transform_val(search[:, :, ::-1])  # bgr->rgb

        if self.norm_out:
            # for efficientnet: rgb norm
            template = self.transorm_norm(template)
            search = self.transorm_norm(search)
        else:
            # for resnet_atrous: rgb->bgr un-norm
            template = np.array(template)[:, :, ::-1].transpose((2, 0, 1)).astype(np.float32)
            search = np.array(search)[:, :, ::-1].transpose((2, 0, 1)).astype(np.float32)
        # self.show_img(template.transpose((1,2,0)), "template", template_bbox)
        # self.show_img(search.transpose((1,2,0)), "search", search_bbox)

        # create labels
        cls, delta, delta_weight, overlap = self.anchor_target(search_bbox, neg)
        fc_label, fc_weight, fc_delta = self.anchor_target.create_fc_labels(search_bbox, neg)

        return {
            'template': template,
            'search': search,
            'label_cls': cls,
            'label_loc': delta,
            'label_loc_weight': delta_weight,
            'label_fc': fc_label,
            'label_fc_weight': fc_weight,
            'label_fc_delta': fc_delta,
            'bbox': search_bbox,
        }

    def _get_sample(self, image, region, exemplar_size, sample_size):
        # get rect box [cx, cy, w, h]
        rect = get_min_max_bbox(region)
        avg_chans = np.mean(image, axis=(0, 1))
        # get samples
        sample, bbox = self.crop_like_SiamFC(image, rect, exemplar_size=exemplar_size,
                                             instanc_size=sample_size, padding=avg_chans)
        return sample, bbox

    def crop_like_SiamFC(self, image, rect, context_amount=0.5, exemplar_size=127, instanc_size=255, padding=(0, 0, 0)):
        target_pos = rect[:2]
        target_size = rect[2:]
        wc_z = target_size[1] + context_amount * sum(target_size)
        hc_z = target_size[0] + context_amount * sum(target_size)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        d_search = (instanc_size - exemplar_size) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad
        xbox = [target_pos[0] - s_x / 2, target_pos[1] - s_x / 2,
                target_pos[0] + s_x / 2, target_pos[1] + s_x / 2]

        s = (instanc_size - 1) / s_x
        c = -s * xbox[0]
        d = -s * xbox[1]
        mapping = np.array([[s, 0, c],
                            [0, s, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (instanc_size, instanc_size),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
        bbox = center2corner(Center(instanc_size // 2, instanc_size // 2, rect[2] * s, rect[3] * s))
        return crop, bbox

    def __len__(self):
        return self.samples_per_epoch

    def show_img(self, image, title="test", box=None):
        cv2.namedWindow(title)
        if box is not None:
            x1, y1, x2, y2 = box
            image = cv2.rectangle(image.copy().astype(np.uint8), (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyWindow(title)


class TrkDataset(Dataset):
    """
    Draw pair-wise images as training samples, including template and search image
    """

    def __init__(self, datasets, p_datasets=None, samples_per_epoch=None):
        super(TrkDataset, self).__init__()

        self.sample_size = 511
        self.exemplar_size = 127
        self.search_size = 255

        self.datasets = datasets
        if samples_per_epoch is not None:
            self.samples_per_epoch = samples_per_epoch
        else:
            self.samples_per_epoch = sum([d.num for d in self.datasets])
        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]
        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.template_aug = Augmentation(shift=4, scale=0.05, blur=0.0, flip=0.0, color=0.2, gray=0.2)
        self.search_aug = Augmentation(shift=64, scale=0.18, blur=0.2, flip=0.01, color=0.2, gray=0.2)

        self.point_targer = PointTarget(stride=8, size=29, center_sampling_radius=0.75)

        self.neg_factor = 0.2

    def __getitem__(self, index):
        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]
        index = random.randint(0, dataset.num-1)

        neg = self.neg_factor > random.random()
        # get one dataset
        if neg:
            template = dataset.get_random_target()
            search = random.choices(self.datasets, self.p_datasets)[0].get_random_target()
        else:
            template, search = dataset.get_positive_pair(index)

        # get image
        template_image = cv2.imread(template[0])  # bgr
        search_image = cv2.imread(search[0])  # bgr

        # get sample and bounding box
        z_img, z_box = self._get_sample(template_image, template[1], self.exemplar_size, self.sample_size)
        x_img, x_box = self._get_sample(search_image, search[1], self.exemplar_size, self.sample_size)

        # augmentation, img:[bgr] bbox:[x1,y1,x2,y2]
        template, template_bbox = self.template_aug(z_img, z_box, self.exemplar_size)
        search, search_bbox = self.search_aug(x_img, x_box, self.search_size)

        # template_z, template_box = self._get_box_global(template_image, template[1], out_sz=224)
        # search_x, search_box = self._get_box_global(search_image, search[1], out_sz=224)

        # img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],  # [0.485, 0.456, 0.406],
        #                     std=[58.395, 57.12, 57.375],  # [0.229, 0.224, 0.225],
        #                     to_rgb=True)
        #
        # template_z = self.imnormalize(template_z, **img_norm_cfg)
        # search_x = self.imnormalize(search_x, **img_norm_cfg)

        cls, reg, centerness = self.point_targer(search_bbox, neg_sample=neg)
        # (torch.cat([-reg[..., :2] + self.point_targer.locations, reg[..., 2:] + self.point_targer.locations],
        #            dim=-1)).numpy()
        return {'template': template.transpose((2, 0, 1)).astype(np.float32),
                'search': search.transpose((2, 0, 1)).astype(np.float32),
                'template_box': np.array(template_bbox).astype(np.float32),
                'search_box': np.array(search_bbox).astype(np.float32),
                'cls_label': cls,
                'reg_label': reg,
                'center_label': centerness}

    def __len__(self):
        return self.samples_per_epoch

    def _get_sample(self, image, region, exemplar_size, sample_size):
        # get rect box [cx, cy, w, h]
        rect = get_min_max_bbox(region)
        avg_chans = np.mean(image, axis=(0, 1))
        # get samples
        sample, bbox = self.crop_like_SiamFC(image, rect, exemplar_size=exemplar_size,
                                             instanc_size=sample_size, padding=avg_chans)
        return sample, bbox

    def crop_like_SiamFC(self, image, rect, context_amount=0.5, exemplar_size=127, instanc_size=255, padding=(0, 0, 0)):
        target_pos = rect[:2]
        target_size = rect[2:]
        wc_z = target_size[1] + context_amount * sum(target_size)
        hc_z = target_size[0] + context_amount * sum(target_size)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        d_search = (instanc_size - exemplar_size) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad
        xbox = [target_pos[0] - s_x / 2, target_pos[1] - s_x / 2,
                target_pos[0] + s_x / 2, target_pos[1] + s_x / 2]

        s = (instanc_size - 1) / s_x
        c = -s * xbox[0]
        d = -s * xbox[1]
        mapping = np.array([[s, 0, c],
                            [0, s, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (instanc_size, instanc_size),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
        bbox = center2corner(Center(instanc_size // 2, instanc_size // 2, rect[2] * s, rect[3] * s))
        return crop, bbox

    # def _get_box_global(self, image, region, out_sz=224):
    #     cxywh = get_min_max_bbox(region)
    #     ground_truth = np.array([cxywh[0] - cxywh[2] / 2, cxywh [1] - cxywh[3] /2, cxywh[2], cxywh[3]],
    #                             dtype=np.float32)
    #     imh, imw = image.shape[:2]
    #     w, h = ground_truth[2:]
    #     sx, sy = imw / out_sz, imh / out_sz
    #     bbox = np.array([ground_truth[0] / sx, ground_truth[1] / sy, w / sx, h / sy], dtype=np.float32)
    #     im = cv2.resize(image, (out_sz, out_sz))
    #     return im, bbox

    def imnormalize(self, img, mean, std, to_rgb=True):
        img = img.astype(np.float32)
        if to_rgb:
            img = img[:, :, ::-1]
            # cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return (img - mean) / std

    def imdenormalize(self, img, mean, std, to_bgr=False):
        img = ((img * std) + mean)
        if to_bgr:
            img = img[:, :, ::-1]
            # cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img


class SapDataset(Dataset):
    """ draw dataset as Siamese ATOM Processing (Sap)"""

    def __init__(self, datasets, p_datasets=None, samples_per_epoch=None):
        super(SapDataset, self).__init__()

        self.sample_size = 576
        self.exemplar_size = 144
        self.search_size = 288

        self.datasets = datasets
        if samples_per_epoch is not None:
            self.samples_per_epoch = samples_per_epoch
        else:
            self.samples_per_epoch = sum([d.num for d in self.datasets])
        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]
        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.template_aug = Augmentation(shift=4, scale=0.05, blur=0.0, flip=0.0, color=0.2, gray=0.2)
        self.search_aug = Augmentation(shift=80, scale=0.18, blur=0.2, flip=0.00, color=0.2, gray=0.2)

        self.img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],  # [0.485, 0.456, 0.406],
                                 std=[58.395, 57.12, 57.375],  # [0.229, 0.224, 0.225],
                                 to_rgb=True)

        self.point_targer = CenterReg(288, stride=8, size=23, cls_sampling_radius=4, box_sampling_radius=0.75)
        self.proposal_params = {'min_iou': 0.1, 'boxes_per_frame': 16, 'sigma_factor': [0.01, 0.05, 0.1, 0.2, 0.3]}

        self.neg_factor = 0.2

    def __getitem__(self, index):
        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]
        index = random.randint(0, dataset.num-1)

        neg = self.neg_factor > random.random()
        # get one dataset
        if neg:
            template = dataset.get_random_target()
            search = random.choices(self.datasets, self.p_datasets)[0].get_random_target()
        else:
            template, search = dataset.get_positive_pair(index)

        # get image
        template_image = cv2.imread(template[0])  # bgr
        search_image = cv2.imread(search[0])  # bgr

        # get sample and bounding box
        z_img, z_box = self._get_sample(template_image, template[1], self.exemplar_size, self.sample_size)
        x_img, x_box = self._get_sample(search_image, search[1], self.exemplar_size, self.sample_size)

        # augmentation, img:[bgr] bbox:[x1,y1,x2,y2]
        template_z, template_bbox = self.template_aug(z_img, z_box, self.exemplar_size)
        search_x, search_bbox = self.search_aug(x_img, x_box, self.search_size)

        proposals_box, proposals_iou, weight_iou = self._generate_proposals(np.array(search_bbox).astype(np.float32),
                                                                            neg_sample=neg)
        cls, reg, cls_weights, reg_weights = self.point_targer(search_bbox, neg_sample=neg)

        template_z = self.imnormalize(template_z, **self.img_norm_cfg)
        search_x = self.imnormalize(search_x, **self.img_norm_cfg)
        return {'template': template_z.transpose((2, 0, 1)).astype(np.float32),
                'search': search_x.transpose((2, 0, 1)).astype(np.float32),
                'template_box': np.array(template_bbox).astype(np.float32),
                'search_box': np.array(search_bbox).astype(np.float32),
                'cls_label': cls,
                'cls_weights': cls_weights,
                'reg_label': reg,
                'reg_weights': reg_weights,
                'test_proposals': proposals_box,
                'proposal_iou': proposals_iou,
                'iou_weight': weight_iou}

    def __len__(self):
        return self.samples_per_epoch

    def _generate_proposals(self, box, neg_sample=False):
        num_proposals = self.proposal_params['boxes_per_frame']
        proposals = np.zeros((num_proposals, 4), dtype=np.float32)
        iou_weight = 1
        gt_iou = np.zeros(num_proposals, dtype=np.float32)
        for i in range(num_proposals):
            proposals[i, :], gt_iou[i] = perturb_box(box, min_iou=self.proposal_params['min_iou'],
                                                     sigma_factor=self.proposal_params['sigma_factor'])
        # Map to [-1, 1]
        gt_iou = gt_iou * 2 - 1
        if neg_sample:
            iou_weight = 0
        return proposals, gt_iou, iou_weight

    def _get_sample(self, image, region, exemplar_size, sample_size):
        # get rect box [cx, cy, w, h]
        rect = get_min_max_bbox(region)
        avg_chans = np.mean(image, axis=(0, 1))
        # get samples
        sample, bbox = self.crop_like_ATOM(image, rect, exemplar_size=exemplar_size,
                                           instanc_size=sample_size, padding=avg_chans)
        return sample, bbox

    def crop_like_ATOM(self, image, rect, exemplar_area_factor=5.0 / 2, exemplar_size=144, instanc_size=288,
                       padding=(0, 0, 0)):
        target_pos = rect[:2]
        target_size = np.array(rect[2:])
        wc_z = np.sqrt(target_size.prod()) * exemplar_area_factor
        hc_z = wc_z
        s_z = np.sqrt(wc_z * hc_z)
        d_search = (instanc_size - exemplar_size) / 2
        pad = d_search / exemplar_size * s_z
        s_x = s_z + 2 * pad
        xbox = [target_pos[0] - s_x / 2, target_pos[1] - s_x / 2,
                target_pos[0] + s_x / 2, target_pos[1] + s_x / 2]

        s = (instanc_size - 1) / (s_x + 1e-12)
        c = -s * xbox[0]
        d = -s * xbox[1]
        mapping = np.array([[s, 0, c],
                            [0, s, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (instanc_size, instanc_size),
                              borderMode=cv2.BORDER_REPLICATE)
        bbox = center2corner(Center(instanc_size // 2, instanc_size // 2, rect[2] * s, rect[3] * s))
        return crop, bbox

    def imnormalize(self, img, mean, std, to_rgb=True):
        img = img.astype(np.float32)
        if to_rgb:
            img = img[:, :, ::-1]
            # cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return (img - mean) / std

    def imdenormalize(self, img, mean, std, to_bgr=False):
        img = ((img * std) + mean)
        if to_bgr:
            img = img[:, :, ::-1]
            # cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img