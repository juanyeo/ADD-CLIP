#!/bin/bash

current_time=$(date "+%Y.%m.%d-%H.%M.%S")

CUDA_VISIBLE_DEVICES=2 taskset -c 16-23 python -m training.main --batch-size=2 --lr=1e-5 --wd=0.1 --epochs=6 --workers=4 \
--model EVA02-CLIP-B-16 --pretrained eva --warmup 1000  --zeroshot-frequency 1 --dataset-type grid_distill  \
--test-type coco_panoptic --train-data /data/clipssl/coco/annotations/instances_train2017.json \
--val-data /data/clipssl/coco/annotations/panoptic_val2017.json \
--embed-path metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTB16.npy --train-image-root /data/clipssl/coco/train2017 \
--val-image-root /data/clipssl/coco/val2017  --cache-dir /data/clipssl/checkpoints/EVA02_CLIP_B_psz16_s8B.pt --log-every-n-steps 50 \
--lock-image --save-frequency 6 --lock-image-unlocked-groups 12 --extract-type="v2" \
--name TEST$current_time --downsample-factor 16 --det-image-size 1024 \
--alpha 0.7