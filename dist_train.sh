


#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --configs /data/code/segmentation/config/change_detection_whole/EPUNet/2016/effb1_bce_woedge.yaml

#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/segmentation/segformer/segformer_b0_1024.yaml

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/segmentation_256_128/deeplabv3/effb1_bce.yaml
wait
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/segmentation_256_128/linknet/effb1_bce.yaml
wait
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/segmentation_256_128/ocrnet/effb1_bce.yaml
wait
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/segmentation_256_128/pspnet/effb1_bce.yaml
wait
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/segmentation_256_128/unetplusplus/effb1_bce.yaml
wait
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/hist_match/train_histmatch/effb1_bce.yaml
wait
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/hist_match/test_histmatch/effb1_bce.yaml



