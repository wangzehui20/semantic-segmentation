import os
import csv


def writecsv(logpath, data):
        firstline = []
        logline = []
        for k, v in data.items():
            if isinstance(v, float):
                v = "%.6f" % (v)
                firstline.append(k)
                logline.append(v)
            elif isinstance(v, str):
                firstline.append(k)
                logline.append(v)
            elif isinstance(v, int):
                firstline.append(k)
                logline.append(v)

        if os.path.isfile(logpath):
            with open(logpath, 'a', newline='')as f:
                csv_write = csv.writer(f, dialect='excel')
                csv_write.writerow(logline)
        else:
            with open(logpath, 'w', newline='')as f:
                csv_write = csv.writer(f, dialect='excel')
                csv_write.writerow(firstline)
                csv_write.writerow(logline)

def data_csv(dir, csv_path):
    img_list = os.listdir(dir)
    for i, name in enumerate(img_list):
        data = dict()
        data[''] = i
        data['name'] = name
        writecsv(csv_path, data)


if __name__ == '__main__':
    data_dir = '/data/data/change_detection/2016/splited_images/train/image'
    csv_path = '/data/data/change_detection/2016/splited_images/train/train.csv'
    data_csv(data_dir, csv_path)