# --------change_detection
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/correct/baseline_effb3_dicebce_scse_160.yaml
wait
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/correct/drit_effb3_dicebce_scse_160.yaml
wait
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/correct/hm_effb3_dicebce_scse_160.yaml
wait
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/correct/rd_effb3_dicebce_scse_160.yaml
wait
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/correct/unit_effb3_dicebce_scse_160.yaml


# # --------update
# wait
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py --configs config/update/cyclegan/train/unet/effb1_dicebce.yaml
# # wait
# # CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/cyclegan/train/ocrnet/hr18_dicebce.yaml
# wait
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/cyclegan/train/swintransformer/upernet_swin-s_dicebce.yaml
