#!/bin/bash

#SBATCH --job-name=tet-62
#SBATCH --output=zhores_out-62.txt
#SBATCH --time=1-12:00:00
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=7000

module rm *
module load python/anaconda3
module load gpu/cuda-11.3
source activate tetradat
conda activate tetradat

srun python3 manager.py --task attack --kind attr --data imagenet --model vgg19 --model_attr vgg16 --opt_sc 8

exit 0
