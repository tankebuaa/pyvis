import torch
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.optim.sgd import SGD
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


if __name__ == '__main__':
    model = [{'params':torch.nn.Parameter(torch.randn(2, 2, requires_grad=True)), 'lr':2e-3},
             {'params':torch.nn.Parameter(torch.randn(2, 2, requires_grad=True)), 'lr':2e-2}]
    optim = SGD(model, 0.1)

    # scheduler_warmup is chained with schduler_steplr
    # scheduler_steplr = ExponentialLR(optim, gamma=0.8)
    # scheduler_warmup = GradualWarmupScheduler(optim, multiplier=1.0, total_epoch=5, after_scheduler=scheduler_steplr)

    lambda1 = lambda epoch: (epoch / 5) if epoch < 5 else\
        0.5 * (math.cos((epoch - 5) / (21 - 5) * math.pi) + 1)
    scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda1)
    # this zero gradient update is needed to avoid a warning message, issue #8.
    optim.zero_grad()
    optim.step()
    scheduler_warmup.step()

    for epoch in range(0, 20):
        print("epoch:", epoch, "LR:", scheduler_warmup.get_lr())
        print(epoch, optim.param_groups[0]['lr'])
        print(epoch, optim.param_groups[1]['lr'])
        scheduler_warmup.step()
        optim.step()    # backward pass (update network)


# print
# epoch: 0 LR: [0.0004, 0.004]
# 0 0.0004
# 0 0.004
# epoch: 1 LR: [0.0008, 0.008]
# 1 0.0008
# 1 0.008
# epoch: 2 LR: [0.0012, 0.012]
# 2 0.0012
# 2 0.012
# epoch: 3 LR: [0.0016, 0.016]
# 3 0.0016
# 3 0.016
# epoch: 4 LR: [0.002, 0.02]
# 4 0.002
# 4 0.02
# epoch: 5 LR: [0.0019807852804032307, 0.019807852804032303]
# 5 0.0019807852804032307
# 5 0.019807852804032303
# epoch: 6 LR: [0.0019238795325112869, 0.019238795325112867]
# 6 0.0019238795325112869
# 6 0.019238795325112867
# epoch: 7 LR: [0.0018314696123025453, 0.018314696123025453]
# 7 0.0018314696123025453
# 7 0.018314696123025453
# epoch: 8 LR: [0.0017071067811865474, 0.017071067811865476]
# 8 0.0017071067811865474
# 8 0.017071067811865476
# epoch: 9 LR: [0.0015555702330196021, 0.015555702330196023]
# 9 0.0015555702330196021
# 9 0.015555702330196023
# epoch: 10 LR: [0.00138268343236509, 0.0138268343236509]
# 10 0.00138268343236509
# 10 0.0138268343236509
# epoch: 11 LR: [0.0011950903220161284, 0.011950903220161284]
# 11 0.0011950903220161284
# 11 0.011950903220161284
# epoch: 12 LR: [0.001, 0.01]
# 12 0.001
# 12 0.01
# epoch: 13 LR: [0.0008049096779838719, 0.008049096779838718]
# 13 0.0008049096779838719
# 13 0.008049096779838718
# epoch: 14 LR: [0.0006173165676349102, 0.006173165676349103]
# 14 0.0006173165676349102
# 14 0.006173165676349103
# epoch: 15 LR: [0.00044442976698039807, 0.00444429766980398]
# 15 0.00044442976698039807
# 15 0.00444429766980398
# epoch: 16 LR: [0.00029289321881345256, 0.0029289321881345253]
# 16 0.00029289321881345256
# 16 0.0029289321881345253
# epoch: 17 LR: [0.00016853038769745467, 0.0016853038769745466]
# 17 0.00016853038769745467
# 17 0.0016853038769745466
# epoch: 18 LR: [7.612046748871326e-05, 0.0007612046748871326]
# 18 7.612046748871326e-05
# 18 0.0007612046748871326
# epoch: 19 LR: [1.921471959676957e-05, 0.0001921471959676957]
# 19 1.921471959676957e-05
# 19 0.0001921471959676957

