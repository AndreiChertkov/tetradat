#!/bin/bash

#SBATCH --job-name=Tet-43
#SBATCH --output=zhores_out-43.txt
#SBATCH --time=2-12:00:00
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=7000

module rm *
module load python/anaconda3
module load gpu/cuda-11.3
source activate tetradat
conda activate tetradat

srun python3 manager.py --task attack_target --kind attr --data imagenet --model vgg16 --model_attr vgg19 --opt_sc 6

exit 0
