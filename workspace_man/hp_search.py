import os
import sys

p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)

import argparse
import cv2
import numpy as np
import torch

from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
from model.man_model_builder import ManModelBuilder
from utils.load_model import load_pretrain
from utils.opbox import get_axis_aligned_bbox, get_min_max_bbox
from tracker.ManTracker import ManTracker
from config import man_cfg as cfg


def parse_range(range_str):
    param = map(float, range_str.split(','))
    return np.arange(*param)


def parse_range_int(range_str):
    param = map(int, range_str.split(','))
    return np.arange(*param)


parser = argparse.ArgumentParser(description='Hyperparamter search')
parser.add_argument('--dataset', type=str, help='dataset name to eval')
parser.add_argument('--gpuid', default=7, type=int)
parser.add_argument('--wi', default='0.25,0.45,0.05', type=parse_range)
parser.add_argument('--lr', default='0.2,0.4,0.05', type=parse_range)

args = parser.parse_args()


def run_tracker(tracker, img, gt, video_name, restart=True):
    frame_counter = 0
    lost_number = 0
    toc = 0
    pred_bboxes = []
    if restart:  # VOT2016 and VOT 2018
        for idx, (img, gt_bbox) in enumerate(video):
            if len(gt_bbox) == 4:
                gt_bbox = [gt_bbox[0], gt_bbox[1],
                           gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                           gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                           gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]
            tic = cv2.getTickCount()
            if idx == frame_counter:
                cx, cy, w, h = get_min_max_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                pred_bboxes.append(1)
            elif idx > frame_counter:
                outputs = tracker.track(img)
                pred_bbox = outputs['bbox']
                overlap = vot_overlap(pred_bbox, gt_bbox,
                                      (img.shape[1], img.shape[0]))
                if overlap > 0:
                    # not lost
                    pred_bboxes.append(pred_bbox)
                else:
                    # lost object
                    pred_bboxes.append(2)
                    frame_counter = idx + 5  # skip 5 frames
                    lost_number += 1
            else:
                pred_bboxes.append(0)
            toc += cv2.getTickCount() - tic
        toc /= cv2.getTickFrequency()
        print('Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
            video_name, toc, idx / toc, lost_number))
        return pred_bboxes
    else:
        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []
        for idx, (img, gt_bbox) in enumerate(video):
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx - w / 2, cy - h / 2, w, h]
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                scores.append(None)
                pred_bboxes.append(pred_bbox)
            else:
                outputs = tracker.track(img)
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
                scores.append(outputs['best_score'])
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
        toc /= cv2.getTickFrequency()
        print('Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            video_name, toc, idx / toc))
        return pred_bboxes, scores, track_times


def _check_and_occupation(video_path, result_path):
    if os.path.isfile(result_path):
        return True
    try:
        if not os.path.isdir(video_path):
            os.makedirs(video_path)
    except OSError as err:
        print(err)

    # with open(result_path, 'w') as f:
    #     f.write('Occ')
    return False


if __name__ == '__main__':
    torch.cuda.set_device(args.gpuid if args.gpuid > 3 else 5)
    mi = 7 - args.gpuid

    snap_shot = './checkpoints/model{}_e19.pth'.format(mi)

    num_search = len(args.wi) * len(args.lr)
    print(snap_shot, ": Total search number: {}".format(num_search))

    # dataset
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../../pysot/testing_dataset', args.dataset)

    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=True)
    # create model
    model = ManModelBuilder(out_ch=256, relu=False).cuda()
    model = load_pretrain(model, snap_shot)
    torch.set_grad_enabled(False)
    # create tracker
    tracker = ManTracker(model)

    # save results
    model_name = snap_shot.split('/')[-1].split('.')[0]
    benchmark_path = os.path.join('hp_search_result', args.dataset)

    # sequences
    seqs = list(range(len(dataset)))
    np.random.shuffle(seqs)
    for idx in seqs:
        video = dataset[idx]  # select video
        video.load_img()  # load images
        # shuffle hp-params
        np.random.shuffle(args.wi)
        np.random.shuffle(args.lr)
        for wi in args.wi:
            for lr in args.lr:
                # update hp-params
                cfg.WINDOW_INFLUENCE = float(wi)
                cfg.SCALE_LR = float(lr)
                # tracker result
                tracker_path = os.path.join(benchmark_path,
                                            (model_name + '_wi-{:.3f}'.format(wi) + '_lr-{:.3f}'.format(lr)))
                if 'VOT2016' == args.dataset or 'VOT2018' == args.dataset or 'VOT2019' == args.dataset:
                    video_path = os.path.join(tracker_path, 'baseline', video.name)
                    result_path = os.path.join(video_path, video.name + '_001.txt')
                    if _check_and_occupation(video_path, result_path):
                        continue
                    pred_bboxes = run_tracker(tracker, video.imgs,
                                              video.gt_traj, video.name, restart=True)
                    with open(result_path, 'w') as f:
                        for x in pred_bboxes:
                            if isinstance(x, int):
                                f.write("{:d}\n".format(x))
                            else:
                                f.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')
                elif 'VOT2018-LT' == args.dataset:
                    video_path = os.path.join(tracker_path, 'longterm', video.name)
                    result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                    if _check_and_occupation(video_path, result_path):
                        continue
                    pred_bboxes, scores, track_times = run_tracker(tracker,
                                                                   video.imgs, video.gt_traj, video.name,
                                                                   restart=False)
                    pred_bboxes[0] = [0]
                    with open(result_path, 'w') as f:
                        for x in pred_bboxes:
                            f.write(','.join([str(i) for i in x]) + '\n')
                    result_path = os.path.join(video_path,
                                               '{}_001_confidence.value'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in scores:
                            f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                    result_path = os.path.join(video_path,
                                               '{}_time.txt'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in track_times:
                            f.write("{:.6f}\n".format(x))
                elif 'GOT-10k' == args.dataset:
                    video_path = os.path.join(tracker_path, video.name)
                    if not os.path.isdir(video_path):
                        os.makedirs(video_path)
                    result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                    pred_bboxes, scores, track_times = run_tracker(tracker, video.imgs,
                                                                   video.gt_traj, video.name, restart=False)
                    with open(result_path, 'w') as f:
                        for x in pred_bboxes:
                            f.write(','.join([str(i) for i in x]) + '\n')
                    result_path = os.path.join(video_path,
                                               '{}_time.txt'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in track_times:
                            f.write("{:.6f}\n".format(x))
                else:
                    result_path = os.path.join(tracker_path, '{}.txt'.format(video.name))
                    if _check_and_occupation(tracker_path, result_path):
                        continue
                    pred_bboxes, _, _ = run_tracker(tracker, video.imgs,
                                                    video.gt_traj, video.name, restart=False)
                    with open(result_path, 'w') as f:
                        for x in pred_bboxes:
                            f.write(','.join([str(i) for i in x]) + '\n')


        # free img
        video.free_img()
