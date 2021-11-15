python -m torch.distributed.launch --nproc_per_node=8 ../train_imagenet.py \
 --config ./configs/config_resnet50.yaml --schemes_file ./schemes/resnet50_M16_0.875.txt --model_dir ./resnet50/resnet50_0.875_M16_flops > ./logs/train-resnet50-0.875-flops.log
