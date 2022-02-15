import torch
import torch.optim
from torch.optim import *
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
import os


class PolyLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, epochs, last_epoch=-1):
        # lambda_G = lambda epoch: (1- epoch*(maxir - minir)/(maxir*epochs))
        if last_epoch != -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        super().__init__(optimizer, lambda epoch: (1 - (epoch / epochs) ** 0.9), last_epoch)


class ReduceLROnPlateauPatch(ReduceLROnPlateau, _LRScheduler):
    def get_lr(self):
        return [ group['lr'] for group in self.optimizer.param_groups ]


class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, epochs, target_lr=0, power=0.9, warmup_factor=1.0 / 5,
                 warmup_epoch=5, warmup_method='linear', last_epoch=-1):
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted "
                "got {}".format(warmup_method))

        self.target_lr = target_lr
        self.epochs = epochs
        self.power = power
        self.warmup_factor = warmup_factor
        self.warmup_epoch = warmup_epoch
        self.warmup_method = warmup_method

        super(WarmupPolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        N = self.epochs - self.warmup_epoch
        T = self.last_epoch - self.warmup_epoch
        if self.last_epoch < self.warmup_epoch:
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            elif self.warmup_method == 'linear':
                alpha = float(self.last_epoch) / self.warmup_epoch
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            else:
                raise ValueError("Unknown warmup type.")
            return [self.target_lr + (base_lr - self.target_lr) * warmup_factor for base_lr in self.base_lrs]
        factor = pow(1 - T / N, self.power)
        return [self.target_lr + (base_lr - self.target_lr) * factor for base_lr in self.base_lrs]


# if __name__ == '__main__':
#     import torch
#     from torchvision.models import resnet18
#     epochs = 60
#     model = resnet18()
#     op = torch.optim.SGD(model.parameters(), 1e-3)
#     sc = WarmupPolyLR(op, epochs=epochs, power=0.9, warmup_epoch=3, warmup_method='linear')
#     lr = []
#     for i in range(epochs):
#         sc.step()
#         print(i, sc.last_epoch, sc.get_lr()[0])
#         lr.append(sc.get_lr()[0])

    # import torch
    # from torchvision.models import resnet18
    # epochs = 60
    # model = resnet18()
    # op = torch.optim.AdamW(model.parameters(), lr=0.001)
    # sc = PolyLR(op, epochs)
    # lr = []
    # for i in range(40):
    #     sc.step()
    #     print(i, sc.last_epoch, sc.get_lr()[0])
    #     lr.append(sc.get_lr()[0])