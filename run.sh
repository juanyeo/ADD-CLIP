#!/bin/bash

current_time=$(date "+%Y.%m.%d-%H.%M.%S")

python -m training.main --save-most-recent --delete-previous-checkpoint --batch-size=1 --lr=1e-5 --wd=0.1 --epochs=6 --workers=4 \
--model ViT-B-16 --pretrained openai --warmup 1000  --zeroshot-frequency 1 --dataset-type grid_distill  \
--test-type coco_panoptic --train-data /shared/s2/lab01/dataset/zeroseg/coco/annotations/instances_train2017.json \
--val-data /shared/s2/lab01/dataset/zeroseg/coco/annotations/panoptic_val2017.json \
--embed-path metadata/coco_panoptic_clip_hand_craft_ViTB16.npy --train-image-root /shared/s2/lab01/dataset/zeroseg/coco/train2017 \
--val-image-root /shared/s2/lab01/dataset/zeroseg/coco/val2017 --log-every-n-steps 50 \
--lock-image --save-frequency 6 --lock-image-unlocked-groups 12 --extract-type="v2" \
--name TEST$current_time --downsample-factor 16 --det-image-size 1024 \
--alpha 0.7