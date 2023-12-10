#!/bin/bash -l

#SBATCH --job-name=tet_demo
#SBATCH --output=zhores_demo_out.txt
#SBATCH --time=0-02:00:00
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=40GB

module purge
module load python/anaconda3
module load gpu/cuda-11.3
eval "$(conda shell.bash hook)"
source activate tetradat

pip install protes==0.3.6

srun python manager.py --data imagenet --task check --kind demo

exit 0