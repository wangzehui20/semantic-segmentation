
# wait
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=294 test.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/segmentation/ocrnet/unet.yaml \
# --model /data/data/semi_compete/models/ocrnet/hr18/checkpoints/best.pth --out /data/data/semi_compete/models/ocrnet/hr18/pred_val



CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=294 test.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/cyclegan/test_fake/unet/effb1_bce.yaml \
--model /data/data/update/models_256_128_cyclegan/test_fake/unet/effb1_bce/checkpoints/best.pth --out /data/data/update/models_256_128_cyclegan/test_fake/unet/effb1_bce/pred

