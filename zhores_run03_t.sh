#!/bin/bash

#SBATCH --job-name=Tet-03
#SBATCH --output=zhores_out-03.txt
#SBATCH --time=3-12:00:00
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=15000

module rm *
module load python/anaconda3
module load gpu/cuda-11.3
source activate tetradat
conda activate tetradat

srun python3 manager.py --task attack_target --kind bs_square --data imagenet --model alexnet

srun python3 manager.py --task attack_target --kind bs_square --data imagenet --model vgg16

srun python3 manager.py --task attack_target --kind bs_square --data imagenet --model vgg19

exit 0
