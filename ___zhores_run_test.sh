#!/bin/bash -l

#SBATCH --job-name=test
#SBATCH --output=zhores_out/zhores_out_test.txt
#SBATCH --time=0-03:00:00
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=15GB

module rm *
module load python/anaconda3
module load gpu/cuda-11.3
source activate tetradat
conda activate tetradat

srun python3 manager.py --data imagenet --task check --kind data
srun python3 manager.py --data imagenet --task check --kind model --model googlenet
srun python3 manager.py --data imagenet --task check --kind model --model inception
srun python3 manager.py --data imagenet --task check --kind model --model mobilenet
srun python3 manager.py --data imagenet --task check --kind model --model resnet
srun python3 manager.py --data imagenet --task check --kind model --model vgg

exit 0