


#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --configs /data/code/segmentation/config/change_detection_whole/EPUNet/2016/effb1_bce_woedge.yaml

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --configs /data/code/segmentation_pseudo/config/pseudo/UnetPlusPlus/effb3_dicebce.yaml

