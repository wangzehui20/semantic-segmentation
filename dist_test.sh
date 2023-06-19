

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=294 test.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/ft_split_test/unet_scse_dicebce_size160.yaml \
# --model /data/data/update/models/ft/unet/effb3_dicebce_scse_size160_lr0001_04/checkpoints/best.pth --out /data/data/update/models/ft/unet/effb3_dicebce_scse_size160_lr0001_04/pred
# wait
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=294 test.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/correct/rd_effb3_dicebce_scse_160.yaml \
--model /data/data/update/models/correct/unet/reinhard/checkpoints/k-ep[47]-0.6857.pth --out /data/data/update/models/correct/unet/reinhard/pred_ep47
# wait
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=294 test.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/ft_split_test/unet_scse_dicebce_size160.yaml \
# --model /data/data/update/models/ft/unet/effb3_dicebce_scse_size160_lr0001_08/checkpoints/best.pth --out /data/data/update/models/ft/unet/effb3_dicebce_scse_size160_lr0001_08/pred
# wait
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=294 test.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/update/ft_split_test/unet_scse_dicebce_size160.yaml \
# --model /data/data/update/models/ft/unet/effb3_dicebce_scse_size160_lr0001/checkpoints/best.pth --out /data/data/update/models/ft/unet/effb3_dicebce_scse_size160_lr0001/pred
# wait
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=294 test.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/change_detection/train/split/swin_dicebce.yaml \
# --model /data/data/change_detection/models/train/split/swintransformer/upernet_swin-s_dicebce/checkpoints/best.pth --out /data/data/change_detection/models/train/split/swintransformer/upernet_swin-s_dicebce/pred