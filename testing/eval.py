import numpy as np


def fast_hist(pred, gt, n_classes):
    mask = (gt >= 0) & (gt < n_classes)
    hist = np.bincount(
        n_classes * gt[mask].astype(int) +
        pred[mask], minlength=n_classes ** 2).reshape(n_classes, n_classes)
    return hist


def cal_fscore(prec, recall, f):
    '''
    f: 
        f0.5_score: f == 0.5
        f1_score: f == 1
        f2_score: f == 2
    '''
    return ((1+f**2) * prec * recall) / (f**2 * (prec + recall))


def cal_score(hist):
    tp = hist[1][1]
    fp = hist[0][1]
    tn = hist[0][0]
    fn = hist[1][0]
    hist_sum = hist.sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / hist_sum
    f1 = cal_fscore(precision, recall, 1)
    return f1, precision, recall


def cal_fwiou(hist):
    freq = np.sum(hist, axis=1) / np.sum(hist)
    iou = np.diag(hist) / (np.sum(hist, axis=1) + np.sum(hist, axis=0) - np.diag(hist))
    fwiou = (freq[freq>0] * iou[freq>0]).sum()
    return fwiou


def cal_kappa(hist):
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa