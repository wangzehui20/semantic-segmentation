
# wait
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=294 test.py --configs /data/code/semantic-segmentation-semi-supervised-learning/config/segmentation/segformer/segformer_b0_cedice.yaml \
--model_path /data/data/semi_compete/models/segformer/b0_cedice/checkpoints/best.pth --out /data/data/semi_compete/models/segformer/b0_cedice/pred

