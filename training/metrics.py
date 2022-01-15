import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules import module
from . import _modules as modules
from . import functional as F
from . import base


class IoU(base.Metric):
    __name__ = "iou"

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None,
                 per_image=False, class_weights=None, drop_empty=False, take_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = modules.Activation(activation, dim=1)
        self.ignore_channels = ignore_channels
        self.per_image = per_image
        self.class_weights = class_weights
        self.drop_empty = drop_empty
        self.take_channels = take_channels

    @torch.no_grad()
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.iou(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
            per_image=self.per_image,
            class_weights=self.class_weights,
            drop_empty=self.drop_empty,
            take_channels=self.take_channels,
        )


class MeanIoU(base.Metric):
    __name__ = "mean_iou"

    def __init__(self, ignore_label=0):
        super().__init__()
        self.eps = 1e-5
        self.intersection = {}
        self.union = {}
        self.ignore_label = ignore_label

    def reset(self):
        self.intersection = {}
        self.union = {}

    @torch.no_grad()
    def __call__(self, prediction, target):
        rng = prediction.shape[1]
        prediction = torch.argmax(torch.nn.functional.softmax(prediction, dim=1), dim=1)
        if len(target.shape)==4:   # soft label
            target = torch.argmax(target, dim=1)
        for index in range(rng):
            # ignore background
            if index == self.ignore_label: continue
            pre_single = torch.zeros(prediction.shape).float()
            pre_single[prediction == index] = 1.

            gt_single = torch.zeros(target.shape).float()
            gt_single[target == index] = 1.

            intersection = (pre_single * gt_single).sum()

            union = (pre_single + gt_single).sum() - intersection
            if (index in self.intersection) is False:
                self.intersection[index] = 0
            if (index in self.union) is False:
                self.union[index] = 0

            self.intersection[index] += intersection.detach()
            self.union[index] += union.detach()

        score = 0
        for (k, v) in self.intersection.items():
            intersection = self.intersection[k]
            union = self.union[k]
            score += (intersection + self.eps) / (union + self.eps)

        return score / (rng-1)


class MicroF1(base.Metric):
    __name__ = "micro_f1"

    def __init__(self, threshold=0.5, activation=None):
        super().__init__()
        self.eps = 1e-5
        self.threshold = threshold
        self.activation = modules.Activation(activation, dim=1)

        self.tp = 0.
        self.gt_count = 0.
        self.pre_count = 0.

    def reset(self):
        self.tp = 0.
        self.gt_count = 0.
        self.pre_count = 0.

    @torch.no_grad()
    def __call__(self, prediction, target):
        # single class
        prediction = self.activation(prediction)

        prediction = (prediction > self.threshold).float()
        self.tp += (prediction * target).sum().detach()
        self.gt_count += target.sum().detach()
        self.pre_count += prediction.sum().detach()

        precision = self.tp / self.pre_count
        recall = self.tp / self.gt_count

        score = 2 * precision * recall / (precision + recall)
        return score


class FWIoU(base.Metric):
    __name__ = 'fw_iou'

    def __init__(self, ignore_label=0):
        super().__init__()
        self.ignore_label = ignore_label

    def fast_hist(self, pred, gt, n_classes):
        mask = (gt >= 0) & (gt < n_classes)
        hist = torch.bincount(
            n_classes * gt[mask].int() +
            pred[mask], minlength=n_classes ** 2).reshape(n_classes, n_classes)
        return hist

    @torch.no_grad()
    def __call__(self, pred, target):
        n_classes = pred.shape[1]
        pred = torch.argmax(torch.nn.functional.softmax(pred, dim=1), dim=1)
        hist = self.fast_hist(pred, target, n_classes).data.cpu().numpy()

        freq = np.sum(hist, axis=1) / np.sum(hist)
        iou = np.diag(hist) / (np.sum(hist, axis=1) + np.sum(axis=0) - np.diag(hist))
        fwiou = (freq[freq > 0] * iou[freq > 0]).sum()
        return fwiou


class ChangeIoU(base.Metric):
    __name__ = 'iou'

    def __init__(self):
        super(ChangeIoU, self).__init__()
        self.fhist = 0.
        self.bhist = 0.
        self.ahist = 0.

    def reset(self):
        self.fhist = 0.
        self.bhist = 0.
        self.ahist = 0.

    def fast_hist(self, pred, gt, n_classes=2):
        mask = (gt >= 0) & (gt < n_classes)
        hist = torch.bincount(
            n_classes * gt[mask].int() +
            pred[mask], minlength=n_classes ** 2).reshape(n_classes, n_classes)
        return hist

    def cal_fseg(self, apre_seg, bmask, chg):
        fseg = chg * apre_seg + bmask * (1 - chg)
        return fseg

    @torch.no_grad()
    def __call__(self, pres, gts):
        [bseg_predict, bedge_predict, _], [aseg_predict, aedge_predict, _], chg = pres
        [[bmask, bedge], [amask, aedge]] = gts
        fseg = self.cal_fseg(aseg_predict, bmask, chg).detach()
        self.fhist += self.fast_hist((fseg > 0.5), amask).detach()
        self.bhist += self.fast_hist((bseg_predict > 0.5), bmask).detach()
        self.ahist += self.fast_hist((aseg_predict > 0.5), amask).detach()
        iou = (torch.diag(self.fhist) / (
                    self.fhist.sum(axis=1) + self.fhist.sum(axis=0) - torch.diag(self.fhist))).detach()
        return iou[-1]
