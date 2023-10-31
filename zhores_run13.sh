#!/bin/bash

#SBATCH --job-name=Tet-13
#SBATCH --output=zhores_out-13.txt
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

srun python3 manager.py --task attack_target --kind attr --data imagenet --model alexnet --model_attr vgg16 --opt_sc 6 --attack_num_max 100

exit 0
