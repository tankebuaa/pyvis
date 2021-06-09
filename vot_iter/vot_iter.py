import sys
import cv2
import os

# del os.environ['MKL_NUM_THREADS']
import torch
import numpy as np
import pickle

from os.path import join

import vot_iter.vot as vot

from model.cosi_model_builder import CoModelBuilder
from tracker.CoSiamRPNTracker import CoSiamRPNTracker
from utils.opbox import get_axis_aligned_bbox, get_min_max_bbox
from utils.load_model import load_pretrain

# modify root
pre_fetch = True
if pre_fetch:
    meta_data = "/media/ssd1/ke_tan/vot2019.pkl"
    with open(meta_data, 'rb') as mf:
        videos = pickle.load(mf)

cfg_root = "/home/ke_tan/STUDY/CODE/Python/benchmark/pyvis/workspace_cosiamrpn"
model_file = join(cfg_root, 'checkpoints/eff_vot/model_e19.pth')


def setup_tracker():
    # create model
    model = CoModelBuilder(backbone='efficientnetb0').cuda()
    model = load_pretrain(model, model_file).cuda().eval()

    # build tracker
    tracker = CoSiamRPNTracker(model)

    # warmup model
    for i in range(10):
        model.template(torch.FloatTensor(1,3,127,127).cuda())
    return tracker


torch.cuda.set_device(5)
torch.set_grad_enabled(False)
tracker = setup_tracker()
handle = vot.VOT("polygon")

if pre_fetch:
    # prefetch image buffer
    base_path = os.path.dirname(handle._image)
    video_name = base_path.split('/')[-2]
    imgs = videos[video_name]


region = handle.region()
try:
    region = np.array([region[0][0][0], region[0][0][1], region[0][1][0], region[0][1][1],
                       region[0][2][0], region[0][2][1], region[0][3][0], region[0][3][1]])
except:
    region = np.array(region)

cx, cy, w, h = get_min_max_bbox(region)

img_file = handle.frame()
if not img_file:
    sys.exit(0)
# with open('/home/ke_tan/STUDY/CODE/Python/benchmark/pyvis/vot_iter/debug.txt', 'w') as f:
#     f.write('Hello, world0!' + img_file)
if pre_fetch:
    im = imgs[img_file]  # HxWxC
else:
    im = cv2.imread(img_file)

# init
target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]

tracker.init(im, gt_bbox_)

while True:
    img_file = handle.frame()
    if not img_file:
        break
    if pre_fetch:
        im = imgs[img_file]  # HxWxC
    else:
        im = cv2.imread(img_file)
    outputs = tracker.track(im)
    pred_bbox = outputs['bbox']
    result = vot.Rectangle(*pred_bbox)
    score = outputs['best_score']
    handle.report(result, score)
