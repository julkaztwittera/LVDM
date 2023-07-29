#!/bin/bash -l
#SBATCH -J t2v_inference
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --account=plgso2023-gpu-a100
#SBATCH --time=0-01:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --output=output_inference.txt
#SBATCH --mem-per-gpu=40G

module load CUDA/11.3.1
module load GCC/11.3.0
module load cuDNN/8.2.1.32-CUDA-11.3.1
module load GCCcore/11.3.0
module load Miniconda3/4.9.2
conda activate lvdm

PROMPT=" "
OUTDIR="results/pororo_t2v_short_1/gt"

CKPT_PATH="/net/pr2/projects/plgrid/plgg_so2020/LVDM/logs/lvdm_t2v_short_1/checkpoints/last.ckpt"
#"models/t2v/model.ckpt"
CONFIG_PATH="configs/lvdm_short/text2video.yaml"
VID_DIR="/net/pr2/projects/plgrid/plgg_so2020/LVDM/datasets/pororo_mp4/test"

srun python scripts/sample_text2video.py \
    --ckpt_path $CKPT_PATH \
    --config_path $CONFIG_PATH \
    --prompt "$PROMPT" \
    --save_dir $OUTDIR \
    --n_samples 1 \
    --batch_size 1 \
    --seed 1 \
    --show_denoising_progress \
    --save_jpg \
    --vid_dir $VID_DIR \
    --return_gt True
