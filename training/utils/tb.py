import csv
import os
from torch.utils.tensorboard import SummaryWriter


def is_dir(dir):
    if not os.path.exists(dir): os.makedirs(dir)


def readcsv(filepath):
    filedir = os.path.dirname(filepath)
    logpath = os.path.join(filedir, 'log')
    is_dir(logpath)
    writer = SummaryWriter(logpath)
    names = []
    with open(filepath, encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i == 0:
                names.extend(line)
            else:
                for j, name in enumerate(names):
                    writer.add_scalar(name, float(line[j]), i)


if __name__ == '__main__':
    csvpath1 = r'/data/data/change_detection_whole/2012/models/EPUNet/effb1_bce/log.csv'
    csvpath2 = r'/data/data/change_detection_whole/2012/models/EPUNet/effb1_bce_woedge/log.csv'

    csvpath = [csvpath1, csvpath2]
    for path in csvpath:
        readcsv(path)