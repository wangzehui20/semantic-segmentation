import os
import csv
import os.path as osp


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


if __name__ == '__main__':
    ratiocsv_path = r'/data/data/landset30/Unet_bifpn/new512_128/building_pixratio.csv'
    topk = [1000]
    for tk in topk:
        c = Csv()
        c.readcsv(ratiocsv_path, tk)
        topkcsv_path = osp.join(osp.dirname(ratiocsv_path), f'pixel_ratio_{tk}.csv')
        c.write(topkcsv_path)