import pandas as pd
import os
import os.path as osp
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
        if '_fake_B' in name and 'heatmap' not in name:
            oripath = os.path.join(oridir, name)
            # name = name.split('_fake_B')[0].replace('2012_merge', '2016_merge') + '.png'
            name = name.split('_fake_B')[0] + '.png'
            dstpath = os.path.join(dstdir, name)
            shutil.copyfile(oripath, dstpath)

def check_dir(dir):
    if not os.path.exists(dir): os.makedirs(dir)

def copy_specific(im_dir, im_dst_dir):
    """
    copy specific images to display
    """
    file_names = ['mask']
    # file_names = ['cyclegan', 'hm', 'reinhard', 'train', 'unit', 'drit']
    # file_names = ['deeplabv3/resnet50_dicebce', 'ocrnet/ocrnet_dicebce', 'pspnet/resnet50_dicebce', 'segformer/b2_dicebce', 'swintransformer/upernet_swin-s_dicebce', 'unet/effb3_dicebce_scse_edge_size160']
    # file_names = ['drit_concat']
    # file_names = ['cyclegan']
    # im_names = ['2018_11325.png', '2018_11328.png', '2018_11344.png', '2018_11946.png', '2018_12445.png', '2018_14070.png', '2018_14625.png'] # update
    im_names = ['2012_merge_24179.png', '2012_merge_24260.png', '2012_merge_25709.png', '2012_merge_26537.png', '2012_merge_27914.png', '2012_merge_28589.png', '2012_merge_28646.png', '2012_merge_28654.png'] # change_detection
    for fname in file_names:
        im_dst_base_dir = osp.join(im_dst_dir, fname)
        check_dir(im_dst_base_dir)
        for iname in im_names:
            im_path = osp.join(im_dir, fname, iname)
            # im_path = osp.join(im_dir, fname, 'pred', iname)
            im_dst_path = osp.join(im_dst_base_dir, iname)
            shutil.copyfile(im_path, im_dst_path)


if __name__ == '__main__':
    # train_dir = r'/data/data/newupdate/256_128/train/image'
    # train_gan_dir = r'/data/data/newupdate/256_128/train/image_gan'
    # test_dir = r'/data/data/newupdate/256_128/test/image'
    # test_gan_dir = r'/data/data/newupdate/256_128/test/image_gan'

    # check_dir(train_gan_dir)
    # check_dir(test_gan_dir)
    # copy_gan(train_dir, test_dir, train_gan_dir, test_gan_dir)


    # fake image
    # real_fake_dir = r'/data/data/cyclegan/results_dir/update/update_cyclegan_cam/test_latest/images'
    # train_fake_dir = r'/data/data/update/256_128/train/image_cyclegan_cam'
    # check_dir(train_fake_dir)
    # copy_fake(real_fake_dir, train_fake_dir)


    # copy specific images
    # im_dir = r'/data/data/update/models/cyclegan'
    # im_dst_dir = r'/data/data/update/256_128/test/diff_model_segmentation_0918'
    im_dir = r'/data/data/change_detection/merge/256_128/2012/split4_1/test'
    im_dst_dir = r'/data/data/change_detection/merge/256_128/2012/display_split'
    copy_specific(im_dir, im_dst_dir)
    

