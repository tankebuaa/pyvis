import numpy as np


class TrackerConfig(object):
    def __init__(self, version='otb'):
        # general elements
        self.name = 'IMSiam'
        self.SCORE_SIZE = 29
        self.STRIDE = 8
        self.OFFSET = 15

        self.CONTEXT_AMOUNT = 0.5
        self.EXEMPLAR_SIZE = 127
        self.INSTANCE_SIZE = 255
        self.PENALTY_K = 0.0

        # OTB 0.45,0.35
        self.WINDOW_INFLUENCE = 0.45
        self.SCALE_LR = 0.25


cfg = TrackerConfig(version='otb')


class BaSiamIoUTrackerConfig(object):
    def __init__(self, version='otb'):
        self.DEBUG = 3
        # general elements
        self.name = 'BaSiamIoU'
        self.SCORE_SIZE = 23
        self.STRIDE = 8
        self.exemplar_area_factor = 5.0 / 2
        self.EXEMPLAR_SIZE = 144
        self.INSTANCE_SIZE = 288
        # spatial-constrained template matching
        self.MASK = True
        # refine module
        self.REFINE = True
        self.BASE_BOX = True
        self.STEP = 0.025
        self.TOPK = 1
        # Background-aware salient map
        self.ERA_TH = 0.6  # *hp
        self.ALPHA = 0.3  # *hp
        self.CO_WEIGHT = 0.35  # *hp
        # baseline parameters
        self.PENALTY_K = 0.1
        self.NMS_TH = 0.3
        self.WINDOW_INFLUENCE = 0.36
        self.SCALE_LR = 0.25
        # self.CHECK_FAILURE = False
        #                                                   $cw                $wi   $sl
        # OTB100:  True, True, True,  0.025, 1, 0.60, 0.30, 0.35, 0.10, 0.30, 0.36, 0.25
        # UAV123:  True, True, True,  0.05,  1, 0.60, 0.30, 0.35, 0.10, 0.30, 0.31, 0.25
        # GOT10k:  True, True, False, 0.025, 1, 0.60, 0.30, 0.20, 0.10, 0.30, 0.41, 0.80
        # VOT2019: True, True, False, 0.025, 3, 0.60, 0.30, 0.35, 0.10, 0.30, 0.36, 0.25

    
basi_cfg = BaSiamIoUTrackerConfig(version='otb')
