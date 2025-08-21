#!/bin/bash
#SBATCH --job-name=RL_QSP
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1  # 确保这一行正确指定了 GPU 数量
#SBATCH --time=10:00:00
#SBATCH --partition=gpu  # 确保分区支持 GPU
#SBATCH --output=run_%j.out
#SBATCH --error=run_%j.err

python runner.py -fromjson ../../../model/eval/7bit_params.json