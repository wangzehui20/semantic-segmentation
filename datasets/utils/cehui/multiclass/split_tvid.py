import random
import os
import csv
from tqdm import tqdm


def split_train_val(seed, ratio, num):
    num_list = [n for n in range(num)]
    random.seed(seed)
    random.shuffle(num_list)
    split_idx = int(len(num_list) * ratio)
    train_list = num_list[:split_idx]
    val_list = num_list[split_idx:]
    return train_list, val_list


def writercsv(csvpath, data):
    firstline = []
    firstline.append('')
    firstline.append('name')
    firstline.append('id')
    for i, (k, v) in tqdm(enumerate(data.items()), total=len(data)):
        logline = []
        logline.append(i)
        logline.append(k)
        logline.append(v)

        if os.path.isfile(csvpath):
            with open(csvpath, 'a', newline='')as f:
                csv_write = csv.writer(f, dialect='excel')
                csv_write.writerow(logline)
        else:
            with open(csvpath, 'w', newline='')as f:
                csv_write = csv.writer(f, dialect='excel')
                csv_write.writerow(firstline)
                csv_write.writerow(logline)


if __name__ == '__main__':
    seed = 0
    ratio = 0.7
    traindir = r'/data/data/multiclass/train/images'
    csvpath = r'/data/data/multiclass/trainval.csv'
    trainval_id = {}
    img_list = os.listdir(traindir)
    train_list, val_list = split_train_val(seed, ratio, len(img_list))
    for i, img in enumerate(img_list):
        # train: 0, val: 1
        trainval_id[img] = 0 if i in train_list else 1
    writercsv(csvpath, trainval_id)