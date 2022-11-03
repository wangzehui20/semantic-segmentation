

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=294 test.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/change_detection/cyclegan/train/pspnet/effb1_dicebce.yaml \
# --model /data/data/change_detection/models/cyclegan/pspnet/effb1_dicebce/checkpoints/best.pth --out /data/data/change_detection/models/cyclegan/pspnet/effb1_dicebce/pred
# wait
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=294 test.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/change_detection/cyclegan/train/ocrnet/hr18_dicebce.yaml \
# --model /data/data/change_detection/models/cyclegan/ocrnet/hr18_dicebce/checkpoints/best.pth --out /data/data/change_detection/models/cyclegan/ocrnet/hr18_dicebce/pred
# wait
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=294 test.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/change_detection/cyclegan/train/deeplabv3/effb1_dicebce.yaml \
# --model /data/data/change_detection/models/cyclegan/deeplabv3/effb1_dicebce/checkpoints/best.pth --out /data/data/change_detection/models/cyclegan/deeplabv3/effb1_dicebce/pred
# wait
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=294 test.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/change_detection/cyclegan/train/segformer/b2_dicebce.yaml \
# --model /data/data/change_detection/models/cyclegan/segformer/b2_dicebce/checkpoints/best.pth --out /data/data/change_detection/models/cyclegan/segformer/b2_dicebce/pred
# wait
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=294 test.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/change_detection/cyclegan/train/pspnet/res50_dicebce.yaml \
--model /data/data/change_detection/models/cyclegan/pspnet/res50_dicebce/checkpoints/best.pth --out /data/data/change_detection/models/cyclegan/pspnet/res50_dicebce/pred
wait
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=294 test.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/cyclegan/train/pspnet/res50_dicebce.yaml \
--model /data/data/update/models/cyclegan/pspnet/res50_dicebce/checkpoints/best.pth --out /data/data/update/models/cyclegan/pspnet/res50_dicebce/pred


# ###
# wait
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=294 test.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/cyclegan/train/deeplabv3/effb1_dicebce.yaml \
# --model /data/data/update/models/cyclegan/deeplabv3/effb1_dicebce/checkpoints/best.pth --out /data/data/update/models/cyclegan/deeplabv3/effb1_dicebce/pred
# wait
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=294 test.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/cyclegan/train/ocrnet/hr18_dicebce.yaml \
# --model /data/data/update/models/cyclegan/ocrnet/effb1_dicebce/checkpoints/best.pth --out /data/data/update/models/cyclegan/ocrnet/effb1_dicebce/pred
# wait
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=294 test.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/cyclegan/train/pspnet/effb1_dicebce.yaml \
# --model /data/data/update/models/cyclegan/pspnet/effb1_dicebce/checkpoints/best.pth --out /data/data/update/models/cyclegan/pspnet/effb1_dicebce/pred
# wait
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=294 test.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/cyclegan/train/segformer/b2_dicebce.yaml \
# --model /data/data/update/models/cyclegan/segformer/b2_dicebce/checkpoints/best.pth --out /data/data/update/models/cyclegan/segformer/b2_dicebce/pred
# wait
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=294 test.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/cyclegan/train/swintransformer/upernet_swin-s_dicebce.yaml \
# --model /data/data/update/models/cyclegan/swintransformer/upernet_swin-s_dicebce/checkpoints/best.pth --out /data/data/update/models/cyclegan/swintransformer/upernet_swin-s_dicebce/pred
# wait
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=294 test.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/cyclegan/train/unet/resnet50_dicebce.yaml \
# --model /data/data/update/models/cyclegan/unet/resnet50_dicebce/checkpoints/best.pth --out /data/data/update/models/cyclegan/unet/resnet50_dicebce/pred