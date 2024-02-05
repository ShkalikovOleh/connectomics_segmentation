#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --account=p_pixel
#SBATCH --job-name=training
#SBATCH --gres=gpu:1
#SBATCH --partition=alpha
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=end,fail
#SBATCH --signal=SIGUSR1@90

module switch release/23.04
module load GCCcore/12.2.0
module load Python/3.10.8
module load CUDA/11.8.0

nvidia-smi

CFG_FILE=$1

source "$CFG_FILE"
export $(cut -d= -f1 "$CFG_FILE")
source $VENV_DIR/bin/activate

python -m connectomics_segmentation.train "${@:2}"
