
# wait
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=294 test.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/segmentation/ocrnet/ocrnet_hr18.yaml \
# --model /data/data/semi_compete/models/ocrnet/hr18/checkpoints/best.pth --out /data/data/semi_compete/models/ocrnet/hr18/pred_val



CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=294 test.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/segmentation/SwinTransformer/upernet_swin-s.yaml \
--model /data/data/semi_compete/models/swinTransform/upernet_swin-s/checkpoints/k-ep[79]-0.7133.pth --out /data/data/semi_compete/models/swinTransform/upernet_swin-s/pred_epoch79
wait
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=294 test.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/segmentation/SwinTransformer/upernet_swin-s.yaml \
--model /data/data/semi_compete/models/swinTransform/upernet_swin-s/checkpoints/k-ep[62]-0.7047.pth --out /data/data/semi_compete/models/swinTransform/upernet_swin-s/pred_epoch62
