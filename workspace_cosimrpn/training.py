import os
import sys

sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))

import warnings
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.optim as optim
from model.cosi_model_builder import CoModelBuilder
from datasets import SrtDataset, ImagenetVID, Got10k, MSCOCO, TrackingNet, LaSOT
from torch.utils.tensorboard import SummaryWriter
from utils.warmup_scheduler import GradualWarmupScheduler
from datasets.sotdata import *


def run(rank, size, seed=None):
    set_seed(seed, rank)
    # init dist
    torch.cuda.set_device(rank)
    if size > 1:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '199300'
        dist.init_process_group(backend="nccl", rank=rank, world_size=size)

    # build model
    backbone_type = 'efficientnetb0'
    model = CoModelBuilder(backbone=backbone_type).cuda().train()

    # init model
    if backbone_type is 'resnet50c3':
        pretrained_backbone = torch.load("../model/pretrained_models/resnet50.model",
                                         map_location=lambda storage, loc: storage.cuda())
        model.backbone.load_state_dict(pretrained_backbone, strict=False)
    for sub_model in [model.man_neck, model.man_head, model.corpn_neck, model.corpn_head]:
        for m in sub_model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # dist model
    if size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[rank],
                                                          find_unused_parameters=True)

    # load sub_dataset and make dataset
    root = '../../Datasets'
    anno = '../../Datasets'
    vid_train = ImagenetVID(name='train',
                            root='../../Datasets/ILSVRC2015/Data/VID',
                            anno='../../Datasets/ILSVRC2015/Annotations/VID',
                            frame_range=50)
    got10k_train = Got10k(name='train',
                      root='../../Datasets/got10k/',
                      anno='../../Datasets/got10k/',
                      frame_range=50)
    trackingnet_train = TrackingNet(name='TrackingNet',
                      root='../../Datasets/',
                      anno='../../Datasets/',
                      frame_range=50)
    coco_train = MSCOCO(name='train2017',
                         root='../../Datasets/coco/images',
                         anno='../../Datasets/coco/annotations')

    lasot_val = LaSOT(name='LaSOT', root=root, anno=anno, frame_range=50)

    # data loader
    normlization = False if backbone_type is 'resnet50c3' else True
    train_dataset = SrtDataset([vid_train, got10k_train, trackingnet_train, coco_train],
                               p_datasets=[1, 1, 1, 1],
                               samples_per_epoch=150000,
                               train=True,
                               normlization=normlization) # (4345, 9335, 1120, 849949)
    val_dataset = SrtDataset([lasot_val], samples_per_epoch=5000,
                             train=False, normlization=normlization)

    # dist dataset
    if size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    # data loader
    batch_size = 16
    train_loader = DataLoader(train_dataset, shuffle=(train_sampler is None),
                              batch_size=batch_size,
                              num_workers=4,
                              pin_memory=True,
                              sampler=train_sampler)

    val_loader = DataLoader(val_dataset, shuffle=False,
                            batch_size=batch_size,
                            num_workers=4,
                            pin_memory=True)

    # optimizer & learn_schedule
    if train_sampler:
        module = model.module
    else:
        module = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    # param_dicts = [
    #     {"params": [p for n, p in module.named_parameters() if "backbone" not in n and p.requires_grad]},
    #     {
    #         "params": [p for n, p in module.named_parameters() if "backbone" in n and p.requires_grad],
    #         "lr": 3e-4,
    #     },
    # ]

    # optimizer = torch.optim.AdamW(param_dicts, lr=3e-5, weight_decay=1e-4)
    # lambda1 = lambda epoch: (epoch / 5) if epoch < 5 else\
    #     0.5 * (math.cos((epoch - 5) / (21 - 5) * math.pi) + 1)
    # lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    base_lr = 2e-2 * size * batch_size / 128
    trainable_params = [
        {'params': filter(lambda x: x.requires_grad,
                          module.backbone.parameters()), 'lr': 0.1 * base_lr},
        {'params': module.man_neck.parameters(), 'lr': base_lr},
        {'params': module.man_head.parameters(), 'lr': base_lr},
        {'params': module.corpn_neck.parameters(), 'lr': base_lr},
        {'params': module.corpn_head.parameters(), 'lr': base_lr}]
    optimizer = optim.SGD(trainable_params, lr=0.02, momentum=0.9, weight_decay=1e-4)

    max_epochs = 20
    warmup_epoches = 5
    after_lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0,
                                          total_epoch=warmup_epoches,
                                          after_scheduler=after_lr_scheduler)
    # lambda1 = lambda epoch: (epoch / warmup_epoches) if epoch < warmup_epoches else\
    #     0.5 * (math.cos((epoch - warmup_epoches) / (max_epochs - warmup_epoches + 1) * math.pi) + 1)
    # lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    # this zero gradient update is needed to avoid a warning message.
    optimizer.zero_grad()
    optimizer.step()
    lr_scheduler.step()

    # load_checkpoint
    resume = None
    if resume:
        checkpoint = torch.load("./checkpoints/model_e1.pth", map_location="cpu")
        model.load_state_dict(checkpoint)

    # tensorboard
    if rank == 0:
        train_idx = -1
        tb_train = SummaryWriter('./log/train')
        tb_val = SummaryWriter('./log/val')

    # train phase
    for ep in range(0, max_epochs):
        if rank == 0:
            print("epoch:", ep, "LR:", optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])
        # one epoch for training
        model.train(mode=True)
        # freeze top layers of backbone
        module.backbone.activate_training_layers(mode=False if ep < 10 else True)

        for idx, data in enumerate(train_loader):
            outputs = model(data)
            loss = outputs['total_loss']
            if is_valid_number(loss.data.item()):
                optimizer.zero_grad()
                loss.backward()
                # clip gradient
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10.0)
                optimizer.step()
            if rank == 0:
                train_idx += 1
                if train_idx % 20 == 0:
                    print("train epoch:{}, idx:{}, total_loss:{:.3f}, man_loss:{:.3f}, ori_cls_loss:{:.3f}, "
                          "co_cls_loss:{:.3f}, loc_loss:{:.3f}".format(ep, idx, loss.item(), outputs['man_loss'].item(),
                            outputs['ori_cls_loss'].item(), outputs['co_cls_loss'].item(), outputs['loc_loss'].item()))
                    tb_train.add_scalar("loss/total_loss", loss.item(), train_idx)
                    tb_train.add_scalar("loss/man_loss", outputs['man_loss'].item(), train_idx)
                    tb_train.add_scalar("loss/ori_cls_loss", outputs['ori_cls_loss'].item(), train_idx)
                    tb_train.add_scalar("loss/co_cls_loss", outputs['co_cls_loss'].item(), train_idx)
                    tb_train.add_scalar("loss/loc_loss", outputs['loc_loss'].item(), train_idx)

        if lr_scheduler is not None:
            lr_scheduler.step()

        # save_checkpoint()
        if rank == 0:
            torch.save({'epoch': ep,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()}, "./checkpoints/model_e{}.pth".format(ep))

        # one epoch for validation
        if (ep + 1) % 2 == 0 and ep > 5 and rank == 0:
            val_idx = train_idx
            model.eval()
            with torch.no_grad():
                for idx, data in enumerate(val_loader):
                    out = model(data)
                    val_idx += 10
                    print(
                        "val epoch:{}, idx:{}, total_loss:{:.3f}, man_loss:{:.3f}, ori_cls_loss:{:.3f}, "
                        "co_cls_loss:{:.3f}, loc_loss:{:.3f}".format(ep, idx, out['total_loss'].item(),
                                                                     out['man_loss'].item(), out['ori_cls_loss'].item(),
                                                                     out['co_cls_loss'].item(),
                                                                     out['loc_loss'].item()))
                    tb_val.add_scalar("loss/total_loss", out['total_loss'].item(), val_idx)
                    tb_val.add_scalar("loss/man_loss", out['man_loss'].item(), val_idx)
                    tb_val.add_scalar("loss/ori_cls_loss", out['ori_cls_loss'].item(), val_idx)
                    tb_val.add_scalar("loss/co_cls_loss", out['co_cls_loss'].item(), val_idx)
                    tb_val.add_scalar("loss/loc_loss", out['loc_loss'].item(), val_idx)


def is_valid_number(x):
    return not (math.isnan(x) or math.isinf(x) or x > 1e4)


def set_seed(seed, rank=0):
    if seed is not None:
        seed += rank
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


def main():
    # distribute training and seed
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,'
    distributed = True
    seed = 4

    # main run task
    if distributed:
        world_size = torch.cuda.device_count()
        torch.multiprocessing.spawn(run, args=(world_size, seed,), nprocs=world_size, join=True)
    else:
        run(rank=0, size=1, seed=seed)


if __name__ == "__main__":
    main()
