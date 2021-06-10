"""
configuration
"""


class ManConfig(object):
    def __init__(self, version='otb'):
        self.name = "Man"
        self.SCORE_SIZE = 25
        self.STRIDE = 8
        self.CONTEXT_AMOUNT = 0.5
        self.EXEMPLAR_SIZE = 127
        self.INSTANCE_SIZE = 255

        self.SCORE_UP = 16
        self.SCALE_STEP = 1.0375
        self.SCALE_MUN = 3
        self.SCALE_PENALTY = 0.9745

        # modify
        self.WINDOW_INFLUENCE = 0.3
        self.SCALE_LR = 0.35
        # ch1024relut_OTB100model4_0.652_wi0.3_lr0.35
        # ch256relut_OTB100model2_0.643_wi0.3_lr0.35
        # ch256reluf_OTB100model0_0.651_2i0.25_lr0.35
        self.mam_visual = True


man_cfg = ManConfig()


class CoSiamRPNConfig(object):
    def __init__(self, version='otb'):
        self.name = "CoSiamRPN"
        self.SCORE_SIZE = 25
        self.STRIDE = 8
        self.CONTEXT_AMOUNT = 0.5
        self.EXEMPLAR_SIZE = 127
        self.INSTANCE_SIZE = 255

        self.ANCHOR_RATIOS = [0.33, 0.5, 1, 2, 3]
        self.ANCHOR_SCALES = [8, ]
        # self. CHECK_FAILURE = True

        self.CO_WEIGHT = 0.4
        self.PENALTY_K = 0.25
        self.WINDOW_INFLUENCE = 0.35
        self.MAN_INFLUENCE = 0.25
        self.SCALE_LR = 0.25
        # efficientnetb0_otb100_0.692_cw_0.4_pk_0.25_wi_0.35_mi_0.25_lr_0.25
        # efficientnetb0_vot2018_0.427_cw_0.4_pk_0.25_wi_0.45_mi_0.35_lr_0.55
        # efficientnetb0_vot2019_0.314_cw_0.4_pk_0.25_wi_0.45_mi_0.30_lr_0.25
        # efficientnetb0_LaSOTfull_0.506_cw_0.4_pk_0.05_wi_0.4_mi_0.7_lr_0.25


cosiamrpn_cfg = CoSiamRPNConfig()


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
        self.BASE_BOX = False
        self.STEP = 0.025
        self.TOPK = 3
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


class LGRTrackerConfig(object):
    def __init__(self, version='otb'):
        self.DEBUG = 3
        # general elements
        self.name = 'LGRT'
        self.SCORE_SIZE = 31
        self.STRIDE = 8
        self.OFFSET = 40
        self.CONTEXT_AMOUNT = 0.5
        self.EXEMPLAR_SIZE = 128
        self.INSTANCE_SIZE = 320

        self.WINDOW_INFLUENCE = 0.2
        self.SCALE_LR = 0.25


lgrt_cfg = LGRTrackerConfig(version='otb')
