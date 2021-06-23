import os
import csv
import numpy as np
import glob
import json
from utils.load_text import load_text
from pycocotools.coco import COCO
import torch


class MSCOCO(object):
    """ The COCO dataset. COCO is an image dataset. Thus, we treat each image as a sequence of length 1.

        Publication:
            Microsoft COCO: Common Objects in Context.
            Tsung-Yi Lin, Michael Maire, Serge J. Belongie, Lubomir D. Bourdev, Ross B. Girshick, James Hays, Pietro Perona,
            Deva Ramanan, Piotr Dollar and C. Lawrence Zitnick
            ECCV, 2014
            https://arxiv.org/pdf/1405.0312.pdf

        Download the images along with annotations from http://cocodataset.org/#download. The root folder should be
        organized as follows.
            - coco_root
                - annotations
                    - instances_train2014.json
                    - instances_train2017.json
                - images
                    - train2014
                    - train2017

        Note: You also have to install the coco pythonAPI from https://github.com/cocodataset/cocoapi.
    """

    def __init__(self, name, root, anno):
        cur_path = os.path.dirname(os.path.realpath(__file__))

        self.root = os.path.join(cur_path, '../../', root, name)  # name train2017
        self.anno = os.path.join(cur_path, '../../', anno, 'instances_{}.json'.format(name))

        # Load the COCO set.
        self.coco_set = COCO(self.anno)

        self.cats = self.coco_set.cats

        self.sequence_list = self._get_sequence_list()
        self.sequence_list.sort()

        self.num = len(self.sequence_list)

    def _get_sequence_list(self):
        ann_list = list(self.coco_set.anns.keys())
        seq_list = [a for a in ann_list if self.coco_set.anns[a]['iscrowd'] == 0]

        return seq_list

    def _get_frames(self, seq_id):
        path = self.coco_set.loadImgs([self.coco_set.anns[seq_id]['image_id']])[0]['file_name']
        return path

    def _get_anno(self, seq_id):
        anno = self.coco_set.anns[seq_id]
        return anno

    def get_sequence_info(self, seq_id):
        anno = self._get_anno(seq_id)
        bbox = np.array(anno['bbox'], dtype=np.float32)
        # mask = torch.Tensor(self.coco_set.annToMask(anno)).unsqueeze(dim=0)
        # valid = (bbox[2] > 10) & (bbox[3] > 10)
        # visible = valid
        return bbox#{'bbox': bbox, 'valid': valid, 'visible': visible}

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        video = self.sequence_list[index]

        image = self._get_frames(video)
        image_path = os.path.join(self.root, image)
        image_anno = self.get_sequence_info(video)
        while not ((image_anno[2] > 4) & (image_anno[3] > 4)):
            image_path, image_anno = self.get_random_target()
        return image_path, image_anno

    def get_positive_pair(self, index):
        return self.get_random_target(index), self.get_random_target(index)

    def __len__(self):
        return self.num
        