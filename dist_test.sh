# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=294 test.py --configs config/water/unet/unet_effb3_lr0005_testasval6000.yaml \
# --model_path /data/data/landset30/models_water/Unet_bifpn/effb3_lr0005_testasval6000/checkpoints/best.pth --out /data/data/landset30/models_water/Unet_bifpn/effb3_lr0005_testasval6000/pred


# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=294 test.py --configs /data/code/segmentation/config/change_detection_whole/EPUNet/2016/effb1_bce_woedge.yaml \
# --model_path /data/data/change_detection_whole/2016/models/EPUNet/effb1_bce_woedge/checkpoints/best.pth --out /data/data/change_detection_whole/2016/models/EPUNet/effb1_bce_woedge/pred
# wait
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=294 test.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/segmentation/segformer/segformer_b0_focaldice.yaml \
--model_path /data/data/semi_compete/models/segformer/b0_focaldice/checkpoints/best.pth --out /data/data/semi_compete/models/segformer/b0_focaldice/pred
# wait
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=294 test.py --configs config/water/unet/unet_effb3_lr0005_testasval4000.yaml \
# --model_path /data/data/landset30/models_water/Unet_bifpn/effb3_lr0005_testasval4000/checkpoints/best.pth --out /data/data/landset30/models_water/Unet_bifpn/effb3_lr0005_testasval4000/pred
# wait
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=294 test.py --configs config/water/ocrnet/ocrnet_hr18_b245_testasval8000_t1_dicebce.yaml \
# --model_path /data/data/landset30/models_water/ocrnet/hr18_b245_testasval8000_t1_dicebce/checkpoints/best.pth --out /data/data/landset30/models_water/ocrnet/hr18_b245_testasval8000_t1_dicebce/pred
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=294 test.py --configs config/water/ocrnet/ocrnet_hr18_rgb_dicebce.yaml \
# --model_path /data/data/landset30/models_water/ocrnet/hr18_rgb_dicebce/checkpoints/best.pth --out /data/data/landset30/models_water/ocrnet/hr18_rgb_dicebce/pred_tta
