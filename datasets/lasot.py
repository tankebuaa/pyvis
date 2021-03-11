import os
import csv
import numpy as np
import glob
import json
from utils.load_text import load_text


class LaSOT(object):
    """ LaSOT dataset.

       Publication:
           LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking
           Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao and Haibin Ling
           CVPR, 2019
           https://arxiv.org/pdf/1809.07845.pdf

       Download the dataset from https://cis.temple.edu/lasot/download.html
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
        cache_file = os.path.join(cur_path, 'meta_data', name + '_train.json')
        if os.path.isfile(cache_file):
            # If available, load the pre-processed cache file containing meta-info for each sequence
            with open(cache_file, 'r') as f:
                labels = json.load(f)

            self.labels = labels
        else:
            # Else process the imagenet annotations and generate the cache file
            self.labels = [self.get_sequence_info(id) for id, _ in enumerate(self.sequence_list)]

            with open(cache_file, 'w') as f:
                json.dump(self.labels, f)

        self.num = len(self.labels)

    def _get_sequence_list(self):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(cur_path, 'lasot_train_split.txt')) as f:
            dir_list = list(csv.reader(f))
        dir_list = [(dir_name[0].split('-')[0] + '/' + dir_name[0] ) for dir_name in dir_list]
        return dir_list

    def get_sequence_info(self, seq_id):
        anno_path = os.path.join(self.root, self.sequence_list[seq_id], 'groundtruth.txt')
        bbox = load_text(anno_path, delimiter=',', dtype=np.float32, backend='pandas')
        # occlusion
        occ_path_list = glob.glob(os.path.join(self.root, self.sequence_list[seq_id], 'full_occlusion.txt'))
        if len(occ_path_list) ==0:
            occ = np.zeros_like(bbox, dtype=np.float32)[:,0]
        else:
            occ = load_text(occ_path_list[0], delimiter=(',', None), dtype=np.float32, backend='pandas').squeeze()
        # out of view
        outview_path_list = glob.glob(os.path.join(self.root, self.sequence_list[seq_id], 'out_of_view.txt'))
        if len(outview_path_list) == 0:
            ov = np.zeros_like(bbox, dtype=np.float32)[:, 0]
        else:
            ov = load_text(outview_path_list[0], delimiter=(',', None), dtype=np.float32, backend='pandas').squeeze()
        valid = np.bitwise_and(occ + ov == 0.0, np.min(bbox[:, 2:], axis=1) > 4)
        visible = valid
        return {'bbox': bbox.tolist(), 'valid': valid.tolist(), 'visible': visible.tolist(), }

    def __len__(self):
        return self.num

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
        frame_file = "{:0{}d}.jpg".format(frame+1, 8)
        image_path = os.path.join(self.root, video, 'img', frame_file)
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
    otb_dataset = LaSOT(name='LaSOT',
                      root='../../Datasets/',
                      anno='../../Datasets/',
                      frame_range=50)
    out1 = otb_dataset.get_random_target()
    out2 = otb_dataset.get_positive_pair(30)
    print('test')
    for i in range(1000000000):
        out1 = otb_dataset.get_random_target()
        out2 = otb_dataset.get_positive_pair(30)
        if out1[1][2] < 4 or out1[1][3] < 4:
            print(out1[1])
    print(len(otb_dataset))
    print('done!')