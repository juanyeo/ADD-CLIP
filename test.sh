#!/bin/bash

NAME=$1
CHECKPOINT=$2
current_time=$(date "+%Y.%m.%d-%H.%M.%S")

CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 torchrun --nnodes 1 --nproc_per_node 8 -m training.main --batch-size=1 \
--model EVA02-CLIP-B-16 --pretrained eva --test-type coco_panoptic --train-data="" \
--val-data /data/clipssl/coco/annotations/panoptic_val2017.json \
--embed-path metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTB16.npy \
--val-image-root /data/clipssl/coco/val2017  --cache-dir $CHECKPOINT --extract-type="v2" \
--name $NAME --downsample-factor 16 --det-image-size 1024