#!/bin/bash -l


# ------------------------------------
# --- Install manually before the run:

# module load python/anaconda3
# conda activate && conda remove --name tetradat --all -y
# conda create --name tetradat python=3.8 -y
# conda activate tetradat
# pip install jupyterlab teneva_opti==0.4.2 torch torchvision matplotlib requests urllib3 torchattacks==3.4.0


# ---------------------------------
# --- How to use this shell script:
# Run this script as "sbatch zhores_run.sh"
# Check status as: "squeue"
# See results in "zhores_out.txt"
# Delete the task as "scancel NUMBER"


# ------------
# --- Options:

#SBATCH --job-name=a.chertkov_tetradat
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --partition gpu
##SBATCH --mem-per-cpu=5000MB
##SBATCH --mem=5GB
#SBATCH --mail-type=ALL
#SBATCH --output=zhores_out.txt


# ----------------
# --- Main script:
module rm *
module load python/anaconda3
module load gpu/cuda-12.0
conda activate tetradat

srun python3 manager.py --task check --kind data --data imagenet

srun python3 manager.py --task check --kind model --data imagenet --model alexnet
srun python3 manager.py --task check --kind model --data imagenet --model vgg16
srun python3 manager.py --task check --kind model --data imagenet --model vgg19

srun python3 manager.py --task attack --kind attr --data imagenet --model alexnet --model_attr vgg16
srun python3 manager.py --task attack --kind attr --data imagenet --model alexnet --model_attr vgg19

srun python3 manager.py --task attack --kind attr --data imagenet --model vgg16 --model_attr alexnet
srun python3 manager.py --task attack --kind attr --data imagenet --model vgg16 --model_attr vgg19

srun python3 manager.py --task attack --kind attr --data imagenet --model vgg19 --model_attr alexnet
srun python3 manager.py --task attack --kind attr --data imagenet --model vgg19 --model_attr vgg16

exit 0
