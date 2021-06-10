import os
import torch
from model.cosi_model_builder import CoModelBuilder
from utils.load_model import load_pretrain
from tracker.CoSiamRPNTracker import CoSiamRPNTracker
import cv2
from glob import glob
import numpy as np
from utils.load_text import load_text
from fvcore.nn import FlopCountAnalysis

# from torchvision.models import resnet50
# from thop import profile
#
# model = resnet50()
# input = torch.randn(1, 3, 224, 224)
# flops, params = profile(model, inputs=(input, ))


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)

        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
            video_name.endswith('mp4'):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield cv2.resize(frame, (640, 360))
            else:
                break
    else:
        images = sorted(glob(os.path.join(video_name, 'img', '*.jp*')))  #
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    # config
    os.environ['CUDA_VISIBLE_DEVICES'] = '5,'
    select = False  # True
    video = os.path.join(os.path.dirname((os.path.abspath(__file__))),
                         '../../../../Datasets/OTB/Basketball')

    # create model
    model = CoModelBuilder().cuda() # backbone='efficientnetb0'

    # load model
    model = load_pretrain(model, os.path.join(os.path.dirname((os.path.abspath(__file__))),
                                              './checkpoints/resnet/model_e19.pth'))
    torch.set_grad_enabled(False)

    # input = (torch.randn(1, 3, 255, 255).cuda(),)
    # flops = FlopCountAnalysis(model, input)
    # print(flops.total())
    # print(sum(p.numel() for p in model.parameters()))

    # build tracker
    tracker = CoSiamRPNTracker(model)

    first_frame = True
    if video:
        video_name = video.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    save_video = False
    if save_video:
        video_file = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 60.0, (640, 360))
    count = 0
    for frame in get_frames(video):
        count = count + 1
        if count < 0:
            continue
        if first_frame:
            if select:
                try:
                    init_rect = cv2.selectROI(video_name, frame, False, False)
                except:
                    exit()
            else:
                gt = load_text(os.path.join(os.path.dirname((os.path.abspath(__file__))),
                                            '../../../../Datasets',  # /got10k
                                            video.split('/')[-2],
                                            video.split('/')[-1],
                                            'groundtruth_rect.txt'),  #
                               delimiter=(',', '\t', None),
                               dtype=np.float32,
                               backend='pandas')
                init_rect = gt[0]
            tic = cv2.getTickCount()
            tracker.init(frame, init_rect)
            print(cv2.getTickFrequency() / (cv2.getTickCount() - tic))
            first_frame = False
            if save_video:
                bbox = list(map(int, init_rect))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                              (0, 255, 0), 3)
                video_file.write(frame)
        else:
            tic = cv2.getTickCount()
            outputs = tracker.track(frame)
            print(round(cv2.getTickFrequency() / (cv2.getTickCount() - tic)), ': ', outputs['best_score'])
            bbox = list(map(int, outputs['bbox']))
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                          (0, 255, 0), 3)
            # if not select:
            #     gtb = gt[count-1]
            #     cv2.rectangle(frame, (gtb[0], gtb[1]),
            #                   (gtb[0] + gtb[2], gtb[1] + gtb[3]),
            #                   (0, 0, 255), 2)
            if save_video:
                video_file.write(frame)
            cv2.imshow(video_name, frame)
            cv2.waitKey(1)

    if save_video:
        video_file.release()


if __name__ == '__main__':
    main()
