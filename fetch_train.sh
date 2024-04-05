#!/bin/bash
# Job name:
#SBATCH --job-name=kanghyun_fetch
#
# Account:
#SBATCH --account=fc_icon
#
# Partition:
#SBATCH --partition=savio3_gpu
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks
#SBATCH --ntasks=1

#Processors per task:
#SBATCH --cpus-per-task=4
#
# QoS option for GPU
#SBATCH --qos=gtx2080_gpu3_normal
#
# Number of GPUs
#SBATCH --gres=gpu:GTX2080TI:2
#
# Wall clock limit:
#SBATCH --time=20:00:00
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kanghyun.ryu@berkeley.edu
#
## Commands to run
module load python/3.10.10
module load cuda/11.2
module load cudnn/8.1.1

python ./fetch_main.py


