import torch
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.optim.sgd import SGD
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
    model = [{'params':torch.nn.Parameter(torch.randn(2, 2, requires_grad=True)), 'lr':5e-4},
             {'params':torch.nn.Parameter(torch.randn(2, 2, requires_grad=True)), 'lr':5e-3}]
    optim = SGD(model, 0.1)

    # scheduler_warmup is chained with schduler_steplr
    scheduler_steplr = ExponentialLR(optim, gamma=0.8)
    scheduler_warmup = GradualWarmupScheduler(optim, multiplier=1.0, total_epoch=5, after_scheduler=scheduler_steplr)

    # this zero gradient update is needed to avoid a warning message, issue #8.
    optim.zero_grad()
    optim.step()
    scheduler_warmup.step()

    for epoch in range(0, 20):
        print("epoch:", epoch, "LR:", scheduler_warmup.get_lr())
        scheduler_warmup.step()
        print(epoch, optim.param_groups[0]['lr'])
        print(epoch, optim.param_groups[1]['lr'])
        optim.step()    # backward pass (update network)


# print
# 0 0.01
# 0 0.02
# 1 0.027999999999999997
# 1 0.055999999999999994
# 2 0.046
# 2 0.092
# 3 0.064
# 3 0.128
# 4 0.08199999999999999
# 4 0.16399999999999998
# 5 0.1
# 5 0.2
# 6 0.01
# 6 0.02
# 7 0.08100000000000002
# 7 0.16200000000000003
# 8 0.0729
# 8 0.1458
# 9 0.06561
# 9 0.13122
# 10 0.05904900000000001
# 10 0.11809800000000002
# 11 0.05314410000000001
# 11 0.10628820000000001
# 12 0.04782969000000001
# 12 0.09565938000000002
# 13 0.04304672100000001
# 13 0.08609344200000002
# 14 0.03874204890000001
# 14 0.07748409780000003
# 15 0.03486784401000001
# 15 0.06973568802000002
# 16 0.031381059609000006
# 16 0.06276211921800001
# 17 0.028242953648100012
# 17 0.056485907296200025
# 18 0.02541865828329001
# 18 0.05083731656658002
# 19 0.02287679245496101
# 19 0.04575358490992202

