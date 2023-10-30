#!/bin/bash

#SBATCH --job-name=tet-12
#SBATCH --output=zhores_out-12.txt
#SBATCH --time=3-00:00:00
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=7000

module rm *
module load python/anaconda3
module load gpu/cuda-11.3
source activate tetradat
conda activate tetradat

srun python3 manager.py --task attack --kind attr --data imagenet --model alexnet --model_attr vgg16 --opt_sc 8

exit 0
