#!/bin/bash
# Job name:
#SBATCH --job-name=kanghyun_jupyter
#
# Account:
#SBATCH --account=fc_icon
#
# Partition:
#SBATCH --partition=savio3
#
#
# Wall clock limit:
#SBATCH --time=02:00:00
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kanghyun.ryu@berkeley.edu
#
## Commands to run
module load python/3.10.10
module load cuda/11.2
module load cudnn/8.1.1

jupyter notebook --no-browser --NotebookApp.allow_origin='*' --NotebookApp.ip='0.0.0.0'


