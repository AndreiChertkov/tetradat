#!/bin/bash

#SBATCH --job-name=tet-tmp
#SBATCH --output=tmp_zhores_out.txt
#SBATCH --time=0-05:00:00
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=30GB

module rm *
module load python/anaconda3
module load gpu/cuda-11.3
source activate tetradat
conda activate tetradat

srun python3 manager.py --task attack_target --kind attr --data imagenet --model mobilenet --model_attr alexnet --attack_num_max 50

exit 0
