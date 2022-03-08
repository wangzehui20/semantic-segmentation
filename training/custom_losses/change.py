import torch.nn as nn
import torch


class SegEdgeLoss(nn.Module):
    def __init__(self, ignore_label=-1, is_train=True):
        super().__init__()
        self.ignore_label = ignore_label
        self.isTrain = is_train
        self.bce = nn.BCELoss()

    def forward(self, pres, gts):
        loss = self.bce(pres[0], gts[0])
        if self.isTrain:
            loss += self.bce(pres[1], gts[1])
        return loss


class ChangeLoss(nn.Module):
    def __init__(self, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.bce=nn.BCEWithLogitsLoss(reduce=False)

    def forward(self, pres, gts, is_trainset=None):
        [bpre_seg, bpre_edge, _], [apre_seg, apre_edge, _], chg = pres
        [[bmask, bedge], [amask, aedge]] = gts
        chg = torch.sigmoid(chg.squeeze())
        if self.is_train:
            bloss = self.bce(bpre_seg.squeeze(), bmask.squeeze().float()).mean(1,2) + self.bce(bpre_edge.squeeze(), bedge.squeeze().float()).mean(1,2)
            aloss = self.bce(apre_seg.squeeze(), amask.squeeze().float()).mean(1,2) + self.bce(apre_edge.squeeze(), aedge.squeeze().float()).mean(1,2)
            bmask = bmask.float()
            bmask[bmask == 0] = -1
            bmask[bmask > 0] = 1
            bmask = bmask * 3
            fseg = chg * apre_seg.squeeze()+ bmask.squeeze() * (1-chg)
            aloss += self.bce(fseg, amask.squeeze().float()).mean(1,2)
            vloss = self.bce(apre_seg.squeeze(), ((torch.sigmoid(fseg)>0.5).squeeze().float())).mean(1,2)

            floss = bloss + aloss*is_trainset + vloss*(1-is_trainset)
        else:
            bmask = bmask.float()
            bmask[bmask == 0] = -1
            bmask[bmask > 0] = 1
            bmask = bmask * 3
            fseg = chg * apre_seg.squeeze() + bmask.squeeze() * (1 - chg)
            floss = self.bce(fseg, amask.squeeze().float()).mean(1,2)

        return floss.mean()


class SoftLabelCrossEntropyLoss(nn.Module):
    def __init__(self,
                 ignore_label: int = -1,
                 reduction : str = 'mean'):
        super().__init__()
        self.ignore_label = ignore_label
        self.reduction = reduction

    def forward(self, pr, gt):
        assert pr.shape == gt.shape
        classes = pr.shape[1]
        log_prob = torch.nn.functional.log_softmax(pr, dim=1)
        if self.ignore_label in range(classes):
            b = pr.shape[0]-1
            gt[:,self.ignore_label,...] = 0
        else:
            b = pr.shape[0]
        if self.reduction == 'mean':
            loss = torch.sum(torch.mul(-log_prob, gt)) / b
        elif self.reduction == 'sum':
            loss = torch.sum(torch.mul(-log_prob, gt))
        return loss


class SoftLabelL1Loss(nn.Module):
    def __init__(self,
                 ignore_label: int = -1,
                 reduction: str = 'mean'):
        super().__init__()
        self.ignore_label = ignore_label
        self.reduction = reduction
        self.l1 = nn.L1Loss()

    def forward(self, pr, gt):
        assert pr.shape == gt.shape
        classes = pr.shape[1]
        log_prob = torch.nn.functional.log_softmax(pr, dim=1)
        if self.ignore_label in range(classes):
            b = pr.shape[0]-1
            gt[:,self.ignore_label,...] = 0
        else:
            b = pr.shape[0]
        if self.reduction == 'mean':
            loss = self.l1(-log_prob, gt) / b
        elif self.reduction == 'sum':
            loss = self.l1(-log_prob, gt)
        return loss


class SoftLabelMSELoss(nn.Module):
    def __init__(self,
                 ignore_label: int = -1,
                 reduction: str = 'mean'):
        super().__init__()
        self.ignore_label = ignore_label
        self.reduction = reduction
        self.mse = nn.MSELoss()

    def forward(self, pr, gt):
        assert pr.shape == gt.shape
        classes = pr.shape[1]
        log_prob = torch.nn.functional.log_softmax(pr, dim=1)
        if self.ignore_label in range(classes):
            b = pr.shape[0]-1
            gt[:,self.ignore_label,...] = 0
        else:
            b = pr.shape[0]
        if self.reduction == 'mean':
            loss = self.mse(-log_prob, gt) / b
        elif self.reduction == 'sum':
            loss = self.mse(-log_prob, gt)
        return loss