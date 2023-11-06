#!/bin/bash

#SBATCH --job-name=tet-tmp
#SBATCH --output=tmp_zhores_out.txt
#SBATCH --time=0-5:00:00
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=30GB

module purge
module load python/anaconda3
module load gpu/cuda-11.3

eval "$(conda shell.bash hook)"

source activate tetradat
# conda activate tetradat

srun python3 manager.py --task attack --kind attr --data imagenet --model mobilenet --model_attr alexnet --attack_num_max 10

exit 0
