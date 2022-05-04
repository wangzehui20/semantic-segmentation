import pandas as pd
import os
import shutil
from tqdm import tqdm


def copy_gan(oridir1, oridir2, dstdir1, dstdir2):
    orinames1 = os.listdir(oridir1)
    for name1 in tqdm(orinames1, total=len(orinames1)):
        if 'bg' not in name1:
            if 'huairou' in name1:
                name2 = name1.replace('2018h', '2019h')
            elif 'shunyi' in name1:
                name2 = name1.replace('2018s', '2019s')
            oripath1 = os.path.join(oridir1, name1)
            oripath2 = os.path.join(oridir2, name2)
            dstpath1 = os.path.join(dstdir1, name1)
            dstpath2 = os.path.join(dstdir2, name2)
            shutil.copyfile(oripath1, dstpath1)
            shutil.copyfile(oripath2, dstpath2)
            if len(os.listdir(dstdir1)) != len(os.listdir(dstdir2)):
                print(oripath1, oripath2)
                break


def copy_fake(oridir, dstdir):
    names = os.listdir(oridir)
    for name in names:
        if '_fake_B' in name:
            oripath = os.path.join(oridir, name)
            name = name.split('_fake_B')[0] + '.png'
            dstpath = os.path.join(dstdir, name)
            shutil.copyfile(oripath, dstpath)

def check_dir(dir):
    if not os.path.exists(dir): os.makedirs(dir)


if __name__ == '__main__':
    train_dir = r'/data/data/newupdate/256_128/train/image'
    train_gan_dir = r'/data/data/newupdate/256_128/train/image_gan'
    test_dir = r'/data/data/newupdate/256_128/test/image'
    test_gan_dir = r'/data/data/newupdate/256_128/test/image_gan'

    check_dir(train_gan_dir)
    check_dir(test_gan_dir)
    copy_gan(train_dir, test_dir, train_gan_dir, test_gan_dir)


    # fake image
    # real_fake_dir = r'/data/code/pytorch-CycleGAN-and-pix2pix/results/maps_cyclegan/test_latest/images'
    # train_fake_dir = r'/data/data/update/256_128/test/image_fake'
    # check_dir(train_fake_dir)
    # copy_fake(real_fake_dir, train_fake_dir)


    

