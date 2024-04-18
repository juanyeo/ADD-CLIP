#!/bin/bash

#SBATCH --job-name=ATTN_ENSEMBLE          # Submit a job named "example"
#SBATCH --nodes=1                    # Using 1 node
#SBATCH --gres=gpu:4                 # Using 1 gpu
#SBATCH --time=7-16:00:00            # 1 hour time limit
#SBATCH --ntasks=4
#SBATCH --mem=100000MB                # Using 10GB CPU Memory
#SBATCH --partition=laal3                # Using "b" partition 
#SBATCH --cpus-per-task=4            # Using 4 maximum processor
#SBATCH --output=./S-%x.%j.out       # Make a log file

eval "$(conda shell.bash hook)"
conda activate clipself

export MASTER_PORT=12800

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

current_time=$(date "+%Y.%m.%d-%H.%M.%S")

srun --cpu-bind=v --accel-bind=gn python -m training.main --save-most-recent --delete-previous-checkpoint --batch-size=1 --lr=1e-5 --wd=0.1 --epochs=6 --workers=4 \
--model ViT-B-16 --pretrained openai --warmup 1000  --zeroshot-frequency 1 --dataset-type grid_distill  \
--test-type coco_panoptic --train-data /shared/s2/lab01/dataset/zeroseg/coco/annotations/instances_train2017.json \
--val-data /shared/s2/lab01/dataset/zeroseg/coco/annotations/panoptic_val2017.json \
--embed-path metadata/coco_panoptic_clip_hand_craft_ViTB16.npy --train-image-root /shared/s2/lab01/dataset/zeroseg/coco/train2017 \
--val-image-root /shared/s2/lab01/dataset/zeroseg/coco/val2017 --log-every-n-steps 50 \
--lock-image --save-frequency 6 --lock-image-unlocked-groups 12 --extract-type="v2" \
--name TEST$current_time --downsample-factor 16 --det-image-size 1024 \
--alpha 0.7