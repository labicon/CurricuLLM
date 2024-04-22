#!/bin/bash
# Job name:
#SBATCH --job-name=kanghyun_visualize
#
# Account:
#SBATCH --account=fc_icon
#
# Partition:
#SBATCH --partition=savio2_gpu
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
#SBATCH --time=00:30:00
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kanghyun.ryu@berkeley.edu
#
## Commands to run
module load python/3.10.10
module load cuda/11.2
module load cudnn/8.1.1

xvfb-run -s "-screen 0 1400x900x24" python visualize_policy.py

