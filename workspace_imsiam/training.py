import os
import warnings
import random
import numpy as np
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.optim as optim
from model.imsi_model_builder import ModelBuilder
from datasets import TrkDataset, ImagenetVID, Got10k, LaSOT, MSCOCO
from torch.utils.tensorboard import SummaryWriter
from utils.warmup_scheduler import GradualWarmupScheduler


def run(rank, size, seed=None):
    set_seed(seed)
    # init dist
    torch.cuda.set_device(rank)
    if size > 0:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '199300'
        dist.init_process_group(backend="nccl", rank=rank, world_size=size)

    # build model
    model = ModelBuilder().cuda().train()

    # init model
    pretrained_backbone = torch.load("model/pretrained_models/resnet50.model",
                                     map_location=lambda storage, loc: storage.cuda())
    model.backbone.load_state_dict(pretrained_backbone, strict=False)
    for sub_model in [model.neck]:
        for m in sub_model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # dist model
    if size > 0:
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
    lasot_train = LaSOT(name='LaSOT',
                      root='../../Datasets/',
                      anno='../../Datasets/',
                      frame_range=50)
    coco_train = MSCOCO(name='train2017',
                         root='../../Datasets/coco/images',
                         anno='../../Datasets/coco/annotations')
    
    train_dataset = TrkDataset([vid_train, got10k_train, lasot_train, coco_train],
                               p_datasets=[1, 1, 1, 2],
                               samples_per_epoch=4000 * 80) # (4345, 9335, 1120, 849949)
    val_dataset = TrkDataset([got10k_train, ], samples_per_epoch=4000)

    # dist dataset
    if size > 0:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = DataLoader(train_dataset, shuffle=(train_sampler is None),
                              batch_size=16,
                              num_workers=2,
                              pin_memory=True,
                              sampler=train_sampler)

    val_loader = DataLoader(val_dataset, shuffle=False,
                            batch_size=16,
                            num_workers=2,
                            pin_memory=True)

    # optimizer, learn_schedule
    if train_sampler:
        module = model.module
    else:
        module = model
    trainable_params = [
        {'params': filter(lambda x: x.requires_grad,
                          module.backbone.parameters()), 'lr': 5e-4},# 5e-4
        {'params': module.neck.parameters(), 'lr': 5e-3},   # 5e-4
        {'params': module.head.parameters(), 'lr': 5e-3}]   # 5e-4
    optimizer = optim.SGD(trainable_params, lr=0.02, momentum=0.9, weight_decay=5e-4)
    after_lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=5, after_scheduler=after_lr_scheduler)

    # this zero gradient update is needed to avoid a warning message, issue #8.
    optimizer.zero_grad()
    optimizer.step()
    lr_scheduler.step()

    max_epochs = 20
    epoch = -1

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
    for ep in range(epoch + 1, max_epochs + 1):
        if rank == 0:
            print("epoch:", ep, "LR:", lr_scheduler.get_lr())
        # one epoch for training
        model.train(mode=True)
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
                    print("train epoch:{}, idx:{}, total_loss:{:.3f}, cls_loss:{:.3f}, box_loss:{:.3f}, "
                          "center_loss:{:.3f}, iou_loss:{:.3f}".format(ep, idx, loss.item(), outputs['cls_loss'].item(),
                            outputs['box_loss'].item(), outputs['centerness_loss'].item(), outputs['iou_loss'].item()))
                    tb_train.add_scalar("loss/total_loss", loss.item(), train_idx)
                    tb_train.add_scalar("loss/cls_loss", outputs['cls_loss'].item(), train_idx)
                    tb_train.add_scalar("loss/centerness_loss", outputs['centerness_loss'].item(), train_idx)
                    tb_train.add_scalar("loss/box_loss", outputs['box_loss'].item(), train_idx)
                    tb_train.add_scalar("loss/iou_loss", outputs['iou_loss'].item(), train_idx)

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
            model.train(mode=False)
            with torch.no_grad():
                for idx, data in enumerate(val_loader):
                    out = model(data)
                    val_idx += 80
                    print(
                        "val epoch:{}, idx:{}, total_loss:{:.3f}, cls_loss:{:.3f}, box_loss:{:.3f}, "
                        "center_loss:{:.3f}, iou_loss:{:.3f}".format(ep, idx, out['total_loss'].item(),
                                                                     out['cls_loss'].item(), out['box_loss'].item(),
                                                                     out['centerness_loss'].item(),
                                                                     out['iou_loss'].item()))
                    tb_val.add_scalar("loss/total_loss", out['total_loss'].item(), val_idx)
                    tb_val.add_scalar("loss/cls_loss", out['cls_loss'].item(), val_idx)
                    tb_val.add_scalar("loss/centerness_loss", out['centerness_loss'].item(), val_idx)
                    tb_val.add_scalar("loss/box_loss", out['box_loss'].item(), val_idx)
                    tb_val.add_scalar("loss/iou_loss", out['iou_loss'].item(), val_idx)


def is_valid_number(x):
    return not (math.isnan(x) or math.isinf(x) or x > 1e4)


def set_seed(seed):
    if seed is not None:
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
    seed = None

    # main run task
    if distributed:
        world_size = torch.cuda.device_count()
        torch.multiprocessing.spawn(run, args=(world_size, seed,), nprocs=world_size, join=True)
    else:
        run(rank=0, size=0, seed=seed)


if __name__ == "__main__":
    main()
