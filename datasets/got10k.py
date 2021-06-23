import os
import csv
import numpy as np
import glob
import json
from utils.load_text import load_text


class Got10k(object):
    """ GOT-10k dataset.

    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """

    def __init__(self, name, root, anno, frame_range):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.root = os.path.join(cur_path, '../../', root, name)
        self.anno = os.path.join(cur_path, '../../', anno, name)

        self.frame_range = frame_range

        # all folders inside the root
        self.sequence_list = self._get_sequence_list()
        self.sequence_list.sort()

        # labels
        cache_file = os.path.join(cur_path, 'meta_data', 'got10k_' + name + '.json')
        if os.path.isfile(cache_file):
            # If available, load the pre-processed cache file containing meta-info for each sequence
            with open(cache_file, 'r') as f:
                labels = json.load(f)

            self.labels = labels
        else:
            # Else process the imagenet annotations and generate the cache file
            self.labels = [self.get_sequence_info(id, version=name) for id, _ in enumerate(self.sequence_list)]

            with open(cache_file, 'w') as f:
                json.dump(self.labels, f)

        self.num = len(self.labels)

    def _get_sequence_list(self):
        with open(os.path.join(self.root, 'list.txt')) as f:
            dir_list = list(csv.reader(f))
        dir_list = [dir_name[0] for dir_name in dir_list]
        return dir_list

    def get_sequence_info(self, seq_id, version='train'):
        if version == 'train':
            anno_path = os.path.join(self.anno, self.sequence_list[seq_id], 'groundtruth.txt')
        else:
            anno_path = os.path.join(self.anno, self.sequence_list[seq_id]+'.txt')
        bbox = load_text(anno_path, delimiter=',', dtype=np.float32, backend='pandas')
        occ_path_list = glob.glob(os.path.join(self.root, self.sequence_list[seq_id], 'absence.label'))
        if len(occ_path_list) ==0:
            occ = np.zeros_like(bbox, dtype=np.float32)[:,0]
        else:
            occ = load_text(occ_path_list[0], delimiter=(',', None), dtype=np.float32, backend='pandas').squeeze()

        valid = np.bitwise_and(occ == 0.0, np.min(bbox[:, 2:], axis=1) > 10)
        visible = valid
        return {'bbox': bbox.tolist(), 'valid': valid.tolist(), 'visible': visible.tolist(), }

    def __len__(self):
        return self.num

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        video = self.sequence_list[index]
        video_label = self.labels[index]
        while sum(video_label['visible']) < 5:
            index = np.random.randint(0, self.num)
            video = self.sequence_list[index]
            video_label = self.labels[index]
        start, end = 0, len(video_label['visible'])

        frame = np.random.randint(start, end)
        while not video_label['valid'][frame - start]:
            frame = np.random.randint(start, end)
        image_path, image_anno = self.get_image_anno(video, video_label, frame)
        return image_path, image_anno

    def get_image_anno(self, video, tracklets, frame):
        frame_file = "{:0{}d}.jpg".format(frame+1, 8)
        image_path = os.path.join(self.root, video, frame_file)
        image_anno = tracklets['bbox'][frame]
        return image_path, image_anno

    def get_positive_pair(self, index):
        video = self.sequence_list[index]
        video_label = self.labels[index]
        while sum(video_label['visible']) < 5:
            index = np.random.randint(0, self.num)
            video = self.sequence_list[index]
            video_label = self.labels[index]
        start, end = 0, len(video_label['visible'])

        template_frame = np.random.randint(start, end)
        while not video_label['valid'][template_frame - start]:
            template_frame = np.random.randint(start, end)

        left = max(template_frame - self.frame_range, start)
        right = min(template_frame + self.frame_range, end-1) + 1
        search_frame = np.random.randint(left, right)
        while not video_label['valid'][search_frame - start]:
            search_frame = np.random.randint(left, right)

        return self.get_image_anno(video, video_label, template_frame), \
            self.get_image_anno(video, video_label, search_frame)