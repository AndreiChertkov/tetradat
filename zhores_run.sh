#!/bin/bash -l


# ------------
# --- Options:

#SBATCH --job-name=a.chertkov_tetradat
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=0-10:00:00
#SBATCH --partition gpu
##SBATCH --mem-per-cpu=1500MB
##SBATCH --mem=5GB
#SBATCH --mail-type=ALL
#SBATCH --output=zhores_out.txt


# ------------------------------------
# --- Install manually before the run:

# module load python/anaconda3
# conda activate && conda remove --name tetradat --all -y
# conda create --name tetradat python=3.8 -y
# conda activate tetradat
# pip install "jax[cpu]==0.4.3" optax teneva==0.13.2 ttopt==0.5.0 protes==0.2.3 torch torchvision snntorch scikit-image matplotlib PyYAML nevergrad requests urllib3


# ----------------
# --- Main script:
module rm *
module load python/anaconda3
module load gpu/cuda-12.0
conda activate tetradat

srun python3 main.py

exit 0


# ---------------------------------
# --- How to use this shell script:
# Run this script as "sbatch zhores_run.sh"
# Check status as: "squeue"
# Delete the task as "scancel NUMBER"
