import csv
import os


class Csv:
    def __init__(self) -> None:
        self.keys = []
        self.content = dict()
        
    def readcsv(self, filepath, topk=0):
        with open(filepath, encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, line in enumerate(reader):
                if i == 0:
                    for key in line:
                        if key not in self.content.keys():
                            self.content[key] = []
                        self.keys.append(key)
                else:
                    for j, value in enumerate(line):
                        self.content[self.keys[j]].append(value)
                if i == topk:
                    break

    def write(self, logpath):
        num = len(self.content[self.keys[0]])
        for i in range(num):
            data = dict()
            for key in self.keys:
                data[key] = self.content[key][i]
            self.writecsv(logpath, data)

    def writecsv(self, logpath, data):
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


def csv_img_mask(root_path, save_train_path, save_val_path):
    names = os.listdir(root_path)
    c = Csv()
    for i, name in enumerate(names):
        data = {}
        data['name'] = name
        if i % 4 ==0:
            c.writecsv(save_val_path, data)
        else:
            c.writecsv(save_train_path, data)

def csv_img(root_path, save_path):
    names = os.listdir(root_path)
    c = Csv()
    for name in names:
        data = {}
        data['name'] = name
        c.writecsv(save_path, data)


if __name__ == '__main__':
    train_path = r'/data/data/semi_compete/clip_integrate/1024_384/labeled_train/image'
    save_train_path = r'/data/data/semi_compete/clip_integrate/1024_384/labeled_train/train.csv'
    save_val_path = r'/data/data/semi_compete/clip_integrate/1024_384/labeled_train/val.csv'

    test_path = r'/data/data/semi_compete/clip_integrate/1024_384/val/image'
    save_test_path = r'/data/data/semi_compete/clip_integrate/1024_384/val/test.csv'

    pseudo_path = r'/data/data/semi_compete/clip_integrate/1024_384/unlabeled_train/image'
    save_pseudo_path = r'/data/data/semi_compete/clip_integrate/1024_384/unlabeled_train/pseudo.csv'

    csv_img_mask(train_path, save_train_path, save_val_path)
    csv_img(test_path, save_test_path)
    csv_img(pseudo_path, save_pseudo_path)


