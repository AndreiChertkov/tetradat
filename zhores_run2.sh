#!/bin/bash


#SBATCH --job-name=tet-2
#SBATCH --output=zhores_out-2.txt
#SBATCH --time=3-00:00:00
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=7000


# -------------------------------
# --- Do manually before the run:

# module avail
# module load python/anaconda3
# conda info --envs
# conda activate && conda remove --name tetradat --all -y
# conda create --name tetradat python=3.8 -y
# conda activate tetradat
# conda list
# pip install teneva_opti==0.5.1 torch==1.12.1+cu113 torchvision==0.13.1+cu113 matplotlib requests urllib3 torchattacks==3.4.0 --extra-index-url https://download.pytorch.org/whl/cu113 && pip install triton
# conda list


# ---------------------------------
# --- How to use this shell script:
# Run this script as "sbatch zhores_run.sh"
# Check status as: "squeue"
# See results in "zhores_out.txt"
# Delete the task as "scancel NUMBER"


# ----------------
# --- Main script:
module rm *
module load python/anaconda3
module load gpu/cuda-11.3
source activate tetradat
conda activate tetradat
pip install teneva_opti==0.5.1 torch==1.12.1+cu113 torchvision==0.13.1+cu113 matplotlib requests urllib3 torchattacks==3.4.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install triton

srun python3 manager.py --task attack --kind attr --data imagenet --model alexnet --model_attr vgg19
srun python3 manager.py --task result --kind attr --data imagenet --model alexnet --model_attr vgg19

srun python3 manager.py --task attack --kind attr --data imagenet --model alexnet --model_attr vgg19 --opt_sc 8
srun python3 manager.py --task result --kind attr --data imagenet --model alexnet --model_attr vgg19 --opt_sc 8

srun python3 manager.py --task attack --kind attr --data imagenet --model alexnet --model_attr vgg19 --opt_sc 6
srun python3 manager.py --task result --kind attr --data imagenet --model alexnet --model_attr vgg19 --opt_sc 6

exit 0
