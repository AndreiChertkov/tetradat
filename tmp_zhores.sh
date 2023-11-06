#!/bin/bash

#SBATCH --job-name=tet-tmp
#SBATCH --output=tmp_zhores_out.txt
#SBATCH --time=0-5:00:00
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=30GB

module rm *
module load python/anaconda3
module load gpu/cuda-11.3

conda activate
conda remove --name tetradat --all -y
conda create --name tetradat python=3.8 -y
conda activate tetradat
pip install teneva_opti==0.5.1 torch==1.12.1+cu113 torchvision==0.13.1+cu113 matplotlib requests urllib3 torchattacks==3.4.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install triton

source activate tetradat
conda activate tetradat

srun python3 manager.py --task attack --kind attr --data imagenet --model mobilenet --model_attr alexnet --attack_num_max 10

exit 0
