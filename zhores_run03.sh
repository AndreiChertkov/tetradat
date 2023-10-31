#!/bin/bash

#SBATCH --job-name=Tet-03
#SBATCH --output=zhores_out-03.txt
#SBATCH --time=2-00:00:00
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=50GB

module rm *
module load python/anaconda3
module load gpu/cuda-11.3
source activate tetradat
conda activate tetradat

srun python3 manager.py --task attack_target --kind bs_square --data imagenet --model alexnet --attack_num_max 100

srun python3 manager.py --task attack_target --kind bs_square --data imagenet --model vgg16 --attack_num_max 100

srun python3 manager.py --task attack_target --kind bs_square --data imagenet --model vgg19 --attack_num_max 100

exit 0
