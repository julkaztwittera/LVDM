#!/bin/bash -l
#SBATCH -J pororo_lvdm_training
#SBATCH --gres=gpu:4
#SBATCH --account=plgso2023-gpu-a100
#SBATCH --time=2-00:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --output=output_autoregression.txt
#SBATCH --mem-per-gpu=40G

module load CUDA/11.3.1
module load GCC/11.3.0
module load cuDNN/8.2.1.32-CUDA-11.3.1
module load GCCcore/11.3.0
module load Miniconda3/4.9.2
conda activate lvdm

PROJ_ROOT="logs"                      # root directory for saving experiment logs
EXPNAME="lvdm_t2v_short_autoregressive_2"          # experiment name 
DATADIR="/net/pr2/projects/plgrid/plgg_so2020/LVDM/datasets/pororo_mp4"  # dataset directory
AEPATH=""
# AEPATH="/net/pr2/projects/plgrid/plgg_so2020/LVDM/logs/lvdm_videoae_pororo/checkpoints/last.ckpt"    # pretrained video autoencoder checkpoint

# CONFIG="configs/lvdm_short/pororo.yaml"
CONFIG="configs/lvdm_short/text2video.yaml"

# run
export TOKENIZERS_PARALLELISM=false
python main.py \
--base $CONFIG \
-t --gpus 0,1,2,3 \
--name $EXPNAME \
--logdir $PROJ_ROOT \
--auto_resume True \
--load_from_checkpoint "/net/pr2/projects/plgrid/plgg_so2020/LVDM/logs/lvdm_t2v_short/checkpoints/model.ckpt" \
lightning.trainer.num_nodes=1 \
data.params.train.params.data_root=$DATADIR \
data.params.validation.params.data_root=$DATADIR \
model.params.first_stage_config.params.ckpt_path=$AEPATH

# --load_from_checkpoint "/net/pr2/projects/plgrid/plgg_so2020/LVDM/logs/lvdm_t2v_short_3/checkpoints/last.ckpt" \
# --load_from_checkpoint "/net/pr2/projects/plgrid/plgg_so2020/LVDM/logs/lvdm_t2v_short/checkpoints/model.ckpt" \
# -------------------------------------------------------------------------------------------------
# commands for multi nodes training
# - use torch.distributed.run to launch main.py
# - set `gpus` and `lightning.trainer.num_nodes`

# For example:

# python -m torch.distributed.run \
#     --nproc_per_node=8 --nnodes=$NHOST --master_addr=$MASTER_ADDR --master_port=1234 --node_rank=$INDEX \
#     main.py \
#     --base $CONFIG \
#     -t --gpus 0,1,2,3,4,5,6,7 \
#     --name $EXPNAME \
#     --logdir $PROJ_ROOT \
#     --auto_resume True \
#     lightning.trainer.num_nodes=$NHOST \
#     data.params.train.params.data_root=$DATADIR \
#     data.params.validation.params.data_root=$DATADIR
