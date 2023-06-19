import cv2
import os
import os.path as osp
import numpy as np

gt_dir = '/data/data/update/256_128/train/display_split/mask_nobg'

# models = os.listdir('/data/data/change_detection/models/color')
models = ['deeplabv3/resnet50_dicebce', 'ocrnet/ocrnet_dicebce', 'pspnet/resnet50_dicebce', 'segformer/b2_dicebce', 'swintransformer/upernet_swin-s_dicebce', 'unet/effb3_dicebce_scse_size160']
for model in models:
    pred_dir = f'/data/data/update/256_128/train/display_split/{model}'
    color_dir = f'/data/data/update/256_128/train/display_color/{model}'
    if not os.path.exists(color_dir):
        os.makedirs(color_dir)
    names = os.listdir(pred_dir)
    for name in names:
        gt_path = osp.join(gt_dir, name)
        pred_path = osp.join(pred_dir, name)
        color_path = osp.join(color_dir, name)
        gt = cv2.imread(gt_path, 0)
        pred = cv2.imread(pred_path, 0)

        gt[gt==255] = 1
        pred[pred==255] = 2
        merge = gt + pred  # 0:TN, 1:FN, 2:FP, 3:TP

        color = np.array([merge] * 3)
        # from IPython import embed; embed()
        # b
        color[0][color[0]!=3] = 0
        color[0][color[0]==3] = 255
        # g
        color[1][color[1]==1] = 255
        color[1][color[1]==2] = 0
        color[1][color[1]==3] = 255
        # r
        color[2][color[2]==1] = 0
        color[2][color[2]==2] = 255
        color[2][color[2]==3] = 255

        cv2.imwrite(color_path, color.transpose(1,2,0))