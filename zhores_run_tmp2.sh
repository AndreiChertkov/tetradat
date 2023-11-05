#!/bin/bash

#SBATCH --job-name=tet-tmp2
#SBATCH --output=zhores_tmp2.txt
#SBATCH --time=0-04:00:00
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=30GB

module rm *
module load python/anaconda3
module load gpu/cuda-11.3
source activate tetradat
conda activate tetradat

srun python3 manager.py --task attack_target --kind attr --data imagenet --model googlenet --model_attr alexnet --attack_num_max 10

exit 0
