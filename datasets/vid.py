import os
import xml.etree.ElementTree as ET
import json
import csv
import numpy as np
import glob
from utils.load_text import load_text


class ImagenetVID(object):
    """ Imagenet VID dataset.

    Publication:
        ImageNet Large Scale Visual Recognition Challenge
        Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy,
        Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei
        IJCV, 2015
        https://arxiv.org/pdf/1409.0575.pdf

    Download the dataset from http://image-net.org/
    """
    def __init__(self, name, root, anno, frame_range):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.name = name
        self.root = os.path.join(cur_path, '../../', root, name)
        self.anno = os.path.join(cur_path, '../../', anno, name)

        self.frame_range = frame_range
        cache_file = os.path.join(cur_path, 'meta_data', 'vid_'+name+'.json')
        if os.path.isfile(cache_file):
            # If available, load the pre-processed cache file containing meta-info for each sequence
            with open(cache_file, 'r') as f:
                sequence_list_dict = json.load(f)

            self.sequence_list = sequence_list_dict
        else:
            # Else process the imagenet annotations and generate the cache file
            self.sequence_list = self._get_sequence_list()

            with open(cache_file, 'w') as f:
                json.dump(self.sequence_list, f)
        self.sequence_list = [sl for sl in self.sequence_list if sl['target_visible'].count(True) > 1]
        self.num = len(self.sequence_list)

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        video = self.sequence_list[index]
        start = video['start_frame']
        end = start + len(video['anno'])
        frame = np.random.randint(start, end)
        while not video['target_visible'][frame - start]:
            frame = np.random.randint(start, end)
        image_path, image_anno = self.get_image_anno(video, frame, frame - start)
        return image_path, image_anno

    def get_image_anno(self, video, frame, anno_idx):
        frame_file = "{:0{}d}.JPEG".format(frame, 6)
        image_path = os.path.join(self.root,
                                  'ILSVRC2015_VID_'+self.name+"_{:04d}".format(video['set_id']),
                                  'ILSVRC2015_'+self.name+"_{:08d}".format(video['vid_id']),
                                  frame_file)
        image_anno = video['anno'][anno_idx]
        return image_path, image_anno

    def get_positive_pair(self, index):
        video = self.sequence_list[index]

        start = video['start_frame']
        end = start + len(video['anno'])

        template_frame = np.random.randint(start, end)
        while not video['target_visible'][template_frame - start]:
            template_frame = np.random.randint(start, end)

        left = max(template_frame - self.frame_range, start)
        right = min(template_frame + self.frame_range, end-1) + 1
        search_frame = np.random.randint(left, right)
        while not video['target_visible'][search_frame - start]:
            search_frame = np.random.randint(left, right)

        return self.get_image_anno(video, template_frame, template_frame-start), \
            self.get_image_anno(video, search_frame, search_frame-start)

    def __len__(self):
        return self.num

    def _get_sequence_list(self):
        base_vid_anno_path = self.anno
        all_sequences = []
        for set in sorted(os.listdir(base_vid_anno_path)):
            set_id = int(set.split('_')[-1])
            print("process: ", set)
            for vid in sorted(os.listdir(os.path.join(base_vid_anno_path, set))):

                vid_id = int(vid.split('_')[-1])
                anno_files = sorted(os.listdir(os.path.join(base_vid_anno_path, set, vid)))

                frame1_anno = ET.parse(os.path.join(base_vid_anno_path, set, vid, anno_files[0]))
                image_size = [int(frame1_anno.find('size/width').text), int(frame1_anno.find('size/height').text)]

                objects = [ET.ElementTree(file=os.path.join(base_vid_anno_path, set, vid, f)).findall('object')
                           for f in anno_files]

                tracklets = {}

                # Find all tracklets along with start frame
                for f_id, all_targets in enumerate(objects):
                    for target in all_targets:
                        tracklet_id = target.find('trackid').text
                        if tracklet_id not in tracklets:
                            tracklets[tracklet_id] = f_id

                for tracklet_id, tracklet_start in tracklets.items():
                    tracklet_anno = []
                    target_visible = []
                    class_name_id = None

                    for f_id in range(tracklet_start, len(objects)):
                        found = False
                        for target in objects[f_id]:
                            if target.find('trackid').text == tracklet_id:
                                if not class_name_id:
                                    class_name_id = target.find('name').text
                                x1 = int(target.find('bndbox/xmin').text)
                                y1 = int(target.find('bndbox/ymin').text)
                                x2 = int(target.find('bndbox/xmax').text)
                                y2 = int(target.find('bndbox/ymax').text)

                                tracklet_anno.append([x1, y1, x2 - x1, y2 - y1])
                                target_visible.append(target.find('occluded').text == '0')

                                found = True
                                break
                        if not found:
                            break

                    new_sequence = {'set_id': set_id, 'vid_id': vid_id, 'class_name': class_name_id,
                                    'start_frame': tracklet_start, 'anno': tracklet_anno,
                                    'target_visible': target_visible, 'image_size': image_size}
                    all_sequences.append(new_sequence)
        return all_sequences



if __name__ == "__main__":
    otb_dataset = ImagenetVID(name='train',
                      root='../../Datasets/ILSVRC2015/Data/VID',
                      anno='../../Datasets/ILSVRC2015/Annotations/VID',
                      frame_range=50)
    out1 = otb_dataset.get_random_target()
    out2 = otb_dataset.get_positive_pair(30)
    for i in range(1000000000):
        print(i)
        out1 = otb_dataset.get_random_target()
        out2 = otb_dataset.get_positive_pair(30)
        if out1[1][2] < 4 or out1[1][3] < 4:
            print(out1[1])
    print(len(otb_dataset))
    print('done!')