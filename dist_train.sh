# --------change_detection
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/cyclegan/train/unet/effb1_dicebce_fakeY.yaml
# wait
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/change_detection/cyclegan/train/unet/effb1_dicebce_fakeY.yaml
# # --------update
# wait
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/cyclegan/train/deeplabv3/effb1_dicebce.yaml
# # wait
# # CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/cyclegan/train/ocrnet/hr18_dicebce.yaml
# wait
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/cyclegan/train/swintransformer/upernet_swin-s_dicebce.yaml
