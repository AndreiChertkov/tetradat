#!/bin/bash

#SBATCH --job-name=tet-01
#SBATCH --output=zhores_out-01.txt
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

srun python3 manager.py --task attack --kind bs_onepixel --data imagenet --model alexnet

srun python3 manager.py --task attack --kind bs_onepixel --data imagenet --model vgg16

srun python3 manager.py --task attack --kind bs_onepixel --data imagenet --model vgg19

exit 0
