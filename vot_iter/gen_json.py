import json
from os import listdir
import glob
from os.path import join
import cv2
import pickle

meta_file = '/home/ke_tan/STUDY/CODE/Python/benchmark/pyvis/datasets/meta_data/vot2019.pkl'
with open(meta_file, 'rb') as mf:
    videos = pickle.load(mf)
tmp = videos['ants1']

data_root = '/home/ke_tan/STUDY/CODE/Matlab/benchmark/vot-toolkit-linux/workspace_vot2019/sequences/'
videos = sorted(listdir(data_root))
dataset = dict()

for video in videos:
    print("process " + video + " ...")
    img_files = sorted(glob.glob(join(data_root, video, 'color/*.jpg')))
    imgs = {}
    for f in img_files:
        img = cv2.imread(f)
        imgs[f] = img
    dataset[video] = imgs
pickle.dump(dataset, open(meta_file, 'wb'))
