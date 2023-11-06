#!/bin/bash

#SBATCH --job-name=8tmp-tet
#SBATCH --output=8tmp_zhores_out.txt
#SBATCH --time=0-3:00:00
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=30GB

module purge
module load python/anaconda3
module load gpu/cuda-11.3
eval "$(conda shell.bash hook)"
source activate tetradat

srun python3 manager.py --task attack --kind attr --data imagenet --model mobilenet --model_attr alexnet --attack_num_max 10 --postfix test8 --opt_d 500 --opt_n 3 --opt_k 100 --opt_k_top 10 --opt_k_gd 100 --opt_lr 0.01 --opt_r 5

exit 0
