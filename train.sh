#!/bin/bash
# Job name:
#SBATCH --job-name=kanghyun
#
# Account:
#SBATCH --account=fc_icon
#
# Partition:
#SBATCH --partition=savio2_1080ti
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks
#SBATCH --ntasks=1

#Processors per task:
#SBATCH --cpus-per-task=8
#
# Number of GPUs
#SBATCH --gres=gpu:4
#
# Wall clock limit:
#SBATCH --time=10:00:00
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kanghyun.ryu@berkeley.edu
#
## Commands to run
module load python/3.10.10
python ./main.py


