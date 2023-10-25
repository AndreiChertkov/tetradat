#!/bin/bash

#SBATCH --job-name=tet-0
#SBATCH --output=zhores_out-0.txt
#SBATCH --time=0-02:00:00
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=7000

module rm *
module load python/anaconda3
module load gpu/cuda-11.3
source activate tetradat
conda activate tetradat

srun python3 manager.py --task check --kind data --data imagenet

srun python3 manager.py --task check --kind model --data imagenet --model alexnet
srun python3 manager.py --task check --kind model --data imagenet --model vgg16
srun python3 manager.py --task check --kind model --data imagenet --model vgg19

exit 0
