import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))

import warnings
import random
import numpy as np
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.optim as optim
from model.man_model_builder import ManModelBuilder
from datasets import SrtDataset, ImagenetVID, Got10k
from torch.utils.tensorboard import SummaryWriter
from utils.warmup_scheduler import GradualWarmupScheduler


parser = argparse.ArgumentParser(description='Man Training')
parser.add_argument('--gpuid', default=7, type=int)
parser.add_argument('--seed', default=0, type=int)
args = parser.parse_args()


def run(rank, size, seed=None):
    set_seed(seed, rank)
    # init dist
    torch.cuda.set_device(rank)
    if size > 1:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '199300'
        dist.init_process_group(backend="nccl", rank=rank, world_size=size)

    # build model1887xzsz
    model = ManModelBuilder(out_ch=256, relu=False).cuda().train()
    pretrained_backbone = torch.load("../model/pretrained_models/resnet50.model",
                                     map_location=lambda storage, loc: storage.cuda())
    model.backbone.load_state_dict(pretrained_backbone, strict=False)
    for sub_model in [model.man_neck, model.man_head]:
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
    got10k_val = Got10k(name='train', root=root + 'got10k/', anno=anno + 'got10k/', frame_range=50)

    # data loader
    normlization = False
    train_dataset = SrtDataset([vid_train,],
                               p_datasets=[1, ],
                               samples_per_epoch=150000,
                               train=True,
                               normlization=normlization)
    val_dataset = SrtDataset([got10k_val], samples_per_epoch=5000,
                             train=False, normlization=normlization)
    # dist dataset
    if size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    batch_size = 32
    train_loader = DataLoader(train_dataset, shuffle=(train_sampler is None),
                              batch_size=batch_size,
                              num_workers=4,
                              pin_memory=True,
                              sampler=train_sampler)

    val_loader = DataLoader(val_dataset, shuffle=False,
                            batch_size=batch_size,
                            num_workers=4,
                            pin_memory=True)

    # optimizer, learn_schedule
    if train_sampler:
        module = model.module
    else:
        module = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    base_lr = 2e-2 * size * batch_size / 128
    trainable_params = [
        {'params': filter(lambda x: x.requires_grad,
                          module.backbone.parameters()), 'lr': 0.1 * base_lr},
        {'params': module.man_neck.parameters(), 'lr': base_lr},
        {'params': module.man_head.parameters(), 'lr': base_lr},]
    optimizer = optim.SGD(trainable_params, lr=0.02, momentum=0.9, weight_decay=5e-4)

    max_epochs = 20
    warmup_epoches = 5
    after_lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0,
                                          total_epoch=warmup_epoches,
                                          after_scheduler=after_lr_scheduler)

    optimizer.zero_grad()
    optimizer.step()
    lr_scheduler.step()

    # tensorboard
    if rank == 0:
        train_idx = -1
        tb_train = SummaryWriter('./log/train{}'.format(args.seed))
        tb_val = SummaryWriter('./log/val{}'.format(args.seed))

    # train phase
    for ep in range(0, max_epochs):
        if rank == 0:
            print("epoch:", ep, "LR:", optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])
        # one epoch for training
        model.train(mode=True)
        # freeze top layers of backbone
        module.backbone.activate_training_layers(mode=False if ep < 10 else True)
        # train one epoch
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
                    print("train epoch:{}, idx:{}, total_loss:{:.3f}".format(ep, idx, loss.item()))
                    tb_train.add_scalar("loss/total_loss", loss.item(), train_idx)
        # update lr
        if lr_scheduler is not None:
            lr_scheduler.step()
        # save_checkpoint()
        if rank == 0:
            torch.save({'epoch': ep,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()}, "./checkpoints/model{}_e{}.pth".format(args.seed, ep))
        # one epoch for validation
        if (ep + 1) % 2 == 0 and ep > 5 and rank == 0:
            val_idx = train_idx
            model.eval()
            with torch.no_grad():
                for idx, data in enumerate(val_loader):
                    out = model(data)
                    val_idx += 30
                    print("val epoch:{}, idx:{}, total_loss:{:.3f}".format(ep, idx, out['total_loss'].item()))
                    tb_val.add_scalar("loss/total_loss", out['total_loss'].item(), val_idx)


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
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpuid)
    seed = args.seed
    distributed = False
    # main run task
    if distributed:
        world_size = torch.cuda.device_count()
        torch.multiprocessing.spawn(run, args=(world_size, seed,), nprocs=world_size, join=True)
    else:
        run(rank=0, size=1, seed=seed)


if __name__ == "__main__":
    main()
    # python training.py --gpuid 0 --seed 0
    # python training.py --gpuid 1 --seed 1
    # python training.py --gpuid 2 --seed 2
    # python training.py --gpuid 3 --seed 3
    # python training.py --gpuid 4 --seed 4
