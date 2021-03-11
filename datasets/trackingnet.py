import os
import xml.etree.ElementTree as ET
import json
import csv
import numpy as np
import glob
from utils.load_text import load_text


class TrackingNet(object):
    def __init__(self, name, root, anno, frame_range):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.root = os.path.join(cur_path, '../../', root, name)
        self.anno = os.path.join(cur_path, '../../', anno, name)

        self.frame_range = frame_range
        # labels
        cache_file = os.path.join(cur_path, 'meta_data', name + '_train.json')
        if os.path.isfile(cache_file):
            # If available, load the pre-processed cache file containing meta-info for each sequence
            with open(cache_file, 'r') as f:
                meta = json.load(f)
            self.sequence_list = meta['sequences']
            self.labels = meta['labels']
        else:
            # all folders inside the root
            self.sequence_list = self._get_sequence_list()
            self.sequence_list.sort()
            # Else process the imagenet annotations and generate the cache file
            self.labels = [self.get_sequence_info(id) for id, _ in enumerate(self.sequence_list)]
            meta = {'sequences': self.sequence_list, 'labels': self.labels}
            with open(cache_file, 'w') as f:
                json.dump(meta, f)

        self.num = len(self.labels)

    def _get_sequence_list(self):
        sequences = []
        dir_list = os.listdir(self.root)
        for stem in dir_list:
            if stem == 'TEST':
                continue
            print(stem)
            seq = os.listdir(os.path.join(self.root, stem, 'frames'))
            for s in seq:
                sequences.append(os.path.join(stem, 'frames', s))
        return sequences

    def __len__(self):
        return self.num

    def get_sequence_info(self, seq_id):
        anno_path = os.path.join(self.root, self.sequence_list[seq_id].replace('frames', 'anno')+'.txt')
        bbox = load_text(anno_path, delimiter=',', dtype=np.float32, backend='numpy')
        # occlusion
        valid = (bbox[:, 0] > 0) & (bbox[:, 1] > 0) & (bbox[:, 2] > 4) & (bbox[:, 3] > 4)
        visible = valid
        return {'bbox': bbox.tolist(), 'valid': valid.tolist(), 'visible': visible.tolist(), }

    def get_random_target(self, index=-1):
        if index == -1:
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
        frame_file = "{}.jpg".format(frame)
        image_path = os.path.join(self.root, video, frame_file)
        image_anno = tracklets['bbox'][frame]
        return image_path, image_anno

    def get_positive_pair(self, index):
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


if __name__ == "__main__":
    otb_dataset = TrackingNet(name='TrackingNet',
                        root='../../Datasets/',
                        anno='../../Datasets/',
                        frame_range=50)
    out1 = otb_dataset.get_random_target()
    out2 = otb_dataset.get_positive_pair(30)
    print("test")
    for i in range(1000000000):
        # print(i)
        out1 = otb_dataset.get_random_target()
        out2 = otb_dataset.get_positive_pair(30)
        if out1[1][2] < 4 or out1[1][3] < 4:
            print(out1[1])
    print(len(otb_dataset))
    print('done!')