import torch
import torch.nn as nn

from . import base
from . import functional as F
from . import _modules as modules
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from .loss import lovasz_losses as L
from .loss import flip_loss
import training.smp_losses as smp_loss


class ftnmt_loss(base.Loss):

    def __init__(self, depth=5, axis=[1, 2, 3], smooth=1.0e-5, **kwargs):
        super().__init__(**kwargs)

        assert depth >= 0, ValueError("depth must be >= 0, aborting...")

        self.smooth = smooth
        self.axis = axis
        self.depth = depth

        if depth == 0:
            self.depth = 1
            self.scale = 1.
        else:
            self.depth = depth
            self.scale = 1. / depth

    def inner_prod(self, prob, label):
        prod = torch.mul(prob, label)
        prod = torch.sum(prod, axis=self.axis)
        return prod

    def tnmt_base(self, preds, labels):
        tpl = self.inner_prod(preds, labels)
        tpp = self.inner_prod(preds, preds)
        tll = self.inner_prod(labels, labels)

        num = tpl + self.smooth
        scale = 1. / self.depth
        denum = 0.0
        for d in range(self.depth):
            a = 2. ** d
            b = -(2. * a - 1.)
            denum = denum + torch.reciprocal(torch.add(a * (tpp + tll), b * tpl) + self.smooth)

        result = torch.mul(num, denum) * scale
        return torch.mean(result, dim=0, keepdim=True)

    def forward(self, preds, labels):
        l1 = self.tnmt_base(preds, labels)
        l2 = self.tnmt_base(1. - preds, 1. - labels)
        result = 0.5 * (l1 + l2)
        return 1. - result


class JaccardLoss(base.Loss):

    def __init__(self, eps=1e-7, activation=None, ignore_channels=None,
                 per_image=False, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = modules.Activation(activation, dim=1)
        self.per_image = per_image
        self.ignore_channels = ignore_channels
        self.class_weights = class_weights

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(
            y_pr, y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
            per_image=self.per_image,
            class_weights=self.class_weights,
        )


class DiceLoss(base.Loss):

    def __init__(self, eps=1e-7, beta=1., activation=None, ignore_channels=None,
                 per_image=False, class_weights=None, drop_empty=False, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = modules.Activation(activation, dim=1)
        self.ignore_channels = ignore_channels
        self.per_image = per_image
        self.class_weights = class_weights
        self.drop_empty = drop_empty

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
            per_image=self.per_image,
            class_weights=self.class_weights,
            drop_empty=self.drop_empty,
        )


class DiceLoss_mask(base.Loss):

    def __init__(self, eps=1e-7, beta=1., activation=None, ignore_channels=None,
                 per_image=False, class_weights=None, drop_empty=False, edge=128,  **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = modules.Activation(activation, dim=1)
        self.ignore_channels = ignore_channels
        self.per_image = per_image
        self.class_weights = class_weights
        self.drop_empty = drop_empty
        self.edge = 128

    def genmask(self, data):
        data[:, :, :self.edge, :] = 0
        data[:, :, :, :self.edge] = 0
        data[:, :, :, -self.edge:] = 0
        data[:, :, -self.edge:, :] = 0
        return data


    def forward(self, y_pr, y_gt):
        y_pr = self.genmask(y_pr)
        y_gt = self.genmask(y_gt)
        y_pr = self.activation(y_pr)
        return 1 - F.f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
            per_image=self.per_image,
            class_weights=self.class_weights,
            drop_empty=self.drop_empty,
        )


class L1Loss(nn.L1Loss, base.Loss):
    pass


class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    def __init__(self, reduce=True, size_average=True):
        super().__init__()


class MSEloss(base.Loss):

    def __init__(self):
        super(MSEloss, self).__init__()
        self.loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)

    def forward(self, pre, gt):
        return self.loss_fn(pre, gt)


class BCELoss(base.Loss):

    def __init__(self, pos_weight=1., neg_weight=1., reduction='mean', label_smoothing=None):
        super().__init__()
        assert reduction in ['mean', None, False]
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, pr, gt):
        # if len(gt.shape) == 3:
        #     gt = gt.unsqueeze(dim=1)
        loss = F.binary_crossentropy(
            pr, gt,
            pos_weight=self.pos_weight,
            neg_weight=self.neg_weight,
            label_smoothing=self.label_smoothing,
        )

        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


class BinaryFocalLoss(base.Loss):
    def __init__(self, alpha=1, gamma=2, class_weights=None, logits=False, reduction='mean', label_smoothing=None):
        super().__init__()
        assert reduction in ['mean', None]
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction
        self.class_weights = class_weights if class_weights is not None else 1.
        self.label_smoothing = label_smoothing

    def forward(self, pr, gt):
        if self.logits:
            bce_loss = nn.functional.binary_cross_entropy_with_logits(pr, gt, reduction='none')
        else:
            bce_loss = F.binary_crossentropy(pr, gt, label_smoothing=self.label_smoothing)

        pt = torch.exp(- bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        focal_loss = focal_loss * torch.tensor(self.class_weights).to(focal_loss.device)

        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()

        return focal_loss


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass


class FocalDiceLoss(base.Loss):

    def __init__(self):
        super().__init__()
        self.focal = BinaryFocalLoss()
        self.dice = DiceLoss(eps=10.)

    def __call__(self, y_pred, y_true):
        return 2 * self.focal(y_pred, y_true) + self.dice(y_pred, y_true)


class FocalFtLoss(base.Loss):

    def __init__(self):
        super().__init__()
        self.focal = BinaryFocalLoss()
        self.ftloss = ftnmt_loss()

    def __call__(self, y_pred, y_true):
        return self.focal(y_pred, y_true) + self.ftloss(y_pred, y_true)


class BCEDiceLoss(base.Loss):

    def __init__(self):
        super().__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()

    def __call__(self, y_pred, y_true):
        return self.bce(y_pred, y_true) + self.dice(y_pred, y_true)


class BBCELoss(base.Loss):

    def __init__(self, reduction='mean', label_smoothing=None):
        super().__init__()
        assert reduction in ['mean', None, False]
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, pr, gt):
        pos = torch.eq(gt, 1).float()
        neg = torch.eq(gt, 0).float()
        num_pos = torch.sum(pos)
        num_neg = torch.sum(neg)
        num_total = num_pos + num_neg
        alpha_pos = num_neg / num_total
        alpha_neg = num_pos / num_total

        loss = F.binary_crossentropy(
            pr, gt,
            pos_weight=alpha_pos,
            neg_weight=alpha_neg,
            label_smoothing=self.label_smoothing,
        )

        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


class Lovaszsigmoid(base.Loss):
    def __init__(self, reduction='mean', ignore_label=255):
        super().__init__()
        assert reduction in ['mean', None, False]
        self.reduction = reduction
        self.ignore_label = ignore_label

    def forward(self, pr, gt):
        # Lovasz need sigmoid input
        # out = torch.sigmoid(pr)
        loss = L.lovasz_softmax(pr, gt, classes=[1], ignore=self.ignore_label)
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss


#-----------------------------------------------------------------------------------------------------------------------
#
#

class Lovaszsoftmax(nn.Module):
    def __init__(self, ignore_label=0):
        super(Lovaszsoftmax, self).__init__()
        self.ignore_label = ignore_label

    def forward(self, pre, gt):
        pre = torch.nn.functional.softmax(pre, dim=1)
        return L.lovasz_softmax(pre, gt, ignore=self.ignore_label)


class CrossEntropy(base.Loss):
    def __init__(self, ignore_label=0, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label)

    def forward(self, pre, gt):
        ph, pw = pre.size(2), pre.size(3)
        h, w = gt.size(1), gt.size(2)
        if ph != h or pw != w:
            pre = torch.nn.functional.upsample(
                    input=pre, size=(h, w), mode='bilinear')

        loss = self.criterion(pre, gt)

        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_label=0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_label
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()



class PixelContrastLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07, max_samples=1024, max_views=11, ignore_index=0):
        super(PixelContrastLoss, self).__init__()

        self.temperature = temperature
        self.base_temperature = base_temperature

        self.ignore_label = ignore_index

        self.max_samples = max_samples
        self.max_views = max_views

    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x > 0 and x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, inputs, targets):
        feats = torch.nn.functional.softmax(inputs, dim=1)
        _, predict = torch.max(feats, 1)
        labels = targets.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)

        loss = self._contrastive(feats_, labels_)

        return loss


class BCE_SCE(base.Loss):
    def __init__(self, ignore_label=0):
        super(BCE_SCE, self).__init__()
        self.ignore_index = ignore_label
        self.dice = smp_loss.DiceLoss(mode='multiclass', ignore_index=ignore_label)
        # self.dice = DiceLoss(activation='logsoftmax', ignore_channels=ignore_label)
        self.sce = smp_loss.SoftCrossEntropyLoss(smooth_factor=0.1, ignore_index=ignore_label)

    def forward(self, inputs, targets):
        return self.dice(inputs, targets)*0.5 + self.sce(inputs, targets)*0.5


class BCE_SCE_Contrast(base.Loss):
    def __init__(self, ignore_label=0):
        super(BCE_SCE_Contrast, self).__init__()
        self.ignore_index = ignore_label
        self.dice = smp_loss.DiceLoss(mode='multiclass', ignore_index=ignore_label)

        self.sce = smp_loss.SoftCrossEntropyLoss(smooth_factor=0.1, ignore_index=ignore_label)
        self.contrast = PixelContrastLoss(ignore_index=ignore_label)

    def forward(self, inputs, targets):
        return self.dice(inputs, targets)*0.45 + self.sce(inputs, targets)*0.45 + self.contrast(inputs, targets)*0.01


class DICE_SCE(base.Loss):
    def __init__(self, ignore_label=-1):
        super(DICE_SCE, self).__init__()
        self.ignore_index = ignore_label
        self.dice = smp_loss.DiceLoss(mode='multiclass', ignore_index=ignore_label)
        # self.dice = DiceLoss(activation='logsoftmax', ignore_channels=ignore_label)
        self.sce = smp_loss.SoftCrossEntropyLoss(smooth_factor=0.1, ignore_index=ignore_label)

    def forward(self, inputs, targets):
        return self.dice(inputs, targets)*0.5 + self.sce(inputs, targets)*0.5


class DICE_BCE(base.Loss):
    def __init__(self, ignore_label=-1):
        super(DICE_BCE, self).__init__()
        self.ignore_index = ignore_label
        self.dice = smp_loss.DiceLoss(mode='binary', ignore_index=ignore_label)
        self.bce = smp_loss.SoftBCEWithLogitsLoss(ignore_index=ignore_label)

    def forward(self, inputs, targets):
        return self.dice(inputs, targets)*0.5 + self.bce(inputs, targets)*0.5


class DICE_SCE_Contrast(base.Loss):
    def __init__(self, ignore_label=-1):
        super(DICE_SCE_Contrast, self).__init__()
        self.ignore_index = ignore_label
        self.dice = smp_loss.DiceLoss(mode='multiclass', ignore_index=ignore_label)

        self.sce = smp_loss.SoftCrossEntropyLoss(smooth_factor=0.1, ignore_index=ignore_label)
        self.contrast = PixelContrastLoss(ignore_index=ignore_label)

    def forward(self, inputs, targets):
        return self.dice(inputs, targets)*0.45 + self.sce(inputs, targets)*0.45 + self.contrast(inputs, targets)*0.01


#-----------------------------------------------------------------------------------------------------------------------
#
#

class Changeloss(nn.Module):
    def __init__(self):
        super(Changeloss, self).__init__()
        self.loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)

    def forward(self, pre, gt):
        channel = int(pre.shape[1] / 2)

        gt_small = 1 - gt

        pre1 = pre[:, :channel] * gt_small
        pre2 = pre[:, channel:] * gt_small
        return self.loss_fn(pre1, pre2)*2


class Changeloss_LEVIR(nn.Module):
    def __init__(self):
        super(Changeloss_LEVIR, self).__init__()
        self.loss_fn = FocalDiceLoss()
        self.activation = nn.Sigmoid()
        self.loss_smm = Changeloss()

    def forward(self, pre, gt):
        channel = int(pre.shape[1] / 2)
        pre_union = self.activation(pre[:, :channel] + pre[:, channel:])*gt
        return self.loss_fn(pre_union, gt) + self.loss_smm(pre, gt)


class Changeloss_new(nn.Module):
    def __init__(self):
        super(Changeloss_new, self).__init__()
        self.loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
        self.loss_dff = torch.nn.MSELoss(reduce=True, size_average=True)

    def forward(self, pre, gt):
        channel = int(pre.shape[1] / 2)

        gt_small = 1 - gt

        pre1 = pre[:, :channel] * gt_small
        pre2 = pre[:, channel:] * gt_small

        pre1_diff = pre[:, :channel] * gt
        pre2_diff = pre[:, channel:] * gt
        loss = self.loss_fn(pre1, pre2) + (1 - self.loss_dff(pre1_diff, pre2_diff))
        # for pre in pres[1:]:
        #     scale = pre.shape[-1]/gt.shape[-1]
        #     gt_resize = nn.functional.interpolate(gt, scale_factor=scale, mode='nearest', align_corners=None)
        #     channel = int(pre.shape[1] / 2)
        #     gt_small = 1 - gt_resize
        #     pre1 = pre[:, :channel] * gt_small
        #     pre2 = pre[:, channel:] * gt_small
        #     pre1_diff = pre[:, :channel] * gt_resize
        #     pre2_diff = pre[:, channel:] * gt_resize
        #     loss +=self.loss_fn(pre1, pre2) + (1 - self.loss_dff(pre1_diff, pre2_diff))

        return loss

class Changeloss_self(nn.Module):
    def __init__(self):
        super(Changeloss_self, self).__init__()
        self.loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
        self.loss_dff = torch.nn.MSELoss(reduce=True, size_average=True)

    def forward(self, pre, gt):
        channel = int(pre.shape[1] / 2)

        gt_small = 1 - gt
        pre1 = pre[:, :channel] * gt_small
        pre2 = pre[:, channel:] * gt_small

        pre1_diff = pre[:, :channel] * gt
        pre2_diff = pre[:, channel:] * gt

        return self.loss_fn(pre1, pre2) + (1 - self.loss_dff(pre1_diff, pre2_diff))


class Changelosswd(nn.Module):
    def __init__(self):
        super(Changelosswd, self).__init__()
        self.ssim_loss = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=7)
        # self.WDLoss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
        # self.loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)

    def forward(self, pre, gt):
        channel = int(pre.shape[1] / 2)

        gt_small = 1 - gt

        pre1 = pre[:, :channel] * gt_small
        pre2 = pre[:, channel:] * gt_small
        p_output = torch.nn.functional.softmax(pre1, dim=1)
        q_output = torch.nn.functional.softmax(pre2, dim=1)
        return 1- self.ssim_loss(p_output, q_output)


class Changelossflip(nn.Module):
    def __init__(self, useactive = False):
        super(Changelossflip, self).__init__()
        self.flip_loss = flip_loss.FLIPLoss()
        self.useactive = useactive

    def forward(self, pre, gt):
        channel = int(pre.shape[1] / 2)

        gt_small = 1 - gt

        pre1 = pre[:, :channel] * gt_small
        pre2 = pre[:, channel:] * gt_small
        if self.useactive:
            pre1 = torch.nn.functional.softmax(pre1, dim=1)
            pre2 = torch.nn.functional.softmax(pre2, dim=1)
        return self.flip_loss(pre1, pre2)


class contrastive_loss(nn.Module):
    def __init__(self, tau=1, normalize=False):
        super(contrastive_loss, self).__init__()
        self.tau = tau
        self.normalize = normalize

    def forward(self, xi, xj):

        x = torch.cat((xi, xj), dim=0)

        is_cuda = x.is_cuda
        sim_mat = torch.mm(x, x.T)
        if self.normalize:
            sim_mat_denom = torch.mm(torch.norm(x, dim=1).unsqueeze(1), torch.norm(x, dim=1).unsqueeze(1).T)
            sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)

        sim_mat = torch.exp(sim_mat / self.tau)

        # no diag because it's not diffrentiable -> sum - exp(1 / tau)
        # diag_ind = torch.eye(xi.size(0) * 2).bool()
        # diag_ind = diag_ind.cuda() if use_cuda else diag_ind

        # sim_mat = sim_mat.masked_fill_(diag_ind, 0)

        # top
        if self.normalize:
            sim_mat_denom = torch.norm(xi, dim=1) * torch.norm(xj, dim=1)
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / sim_mat_denom / self.tau)
        else:
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / self.tau)

        sim_match = torch.cat((sim_match, sim_match), dim=0)

        norm_sum = torch.exp(torch.ones(x.size(0)) / self.tau)
        norm_sum = norm_sum.cuda() if is_cuda else norm_sum
        loss = torch.mean(-torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum)))

        return loss


#-----------------------------------------------------------------------------------------------------------------------
#
# custom loss

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
    def __init__(self, is_train=True, is_label=0):
        super().__init__()
        self.is_train = is_train
        self.is_label = is_label
        self.bce = nn.BCELoss()

    def forward(self, pres, gts):
        [bpre_seg, bpre_edge, _], [apre_seg, apre_edge, _], chg = pres
        [[bmask, bedge], [amask, aedge]] = gts
        if self.is_train:
            bloss = self.bce(bpre_seg, bmask) + self.bce(bpre_edge, bedge)
            aloss = self.bce(apre_seg, amask) + self.bce(apre_edge, aedge)
            # bmask_ = bmask.float().clone()
            # bmask_[bmask_ == 0] = -1
            # bmask_[bmask_ > 0] = 1
            # bmask_ = bmask_ * 3
            fseg = chg * apre_seg+ bmask * (1-chg)
            aloss += self.bce(fseg, amask)
            vloss = self.bce(apre_seg, (fseg>0).float())

            floss = bloss + aloss*self.is_label + vloss*(1-self.is_label)
        else:
            bmask = bmask.float()
            bmask[bmask == 0] = -1
            bmask[bmask > 0] = 1
            bmask = bmask * 3
            fseg = chg * apre_seg + bmask * (1 - chg)
            floss = self.bce(fseg, amask)

        return floss
