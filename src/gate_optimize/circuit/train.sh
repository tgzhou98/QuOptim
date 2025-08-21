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

module load miniforge3/24.11

. /data/apps/miniforge3/etc/profile.d/conda.sh

conda activate rlenv
export http_proxy=http://10.244.6.36:8080
export https_proxy=http://10.244.6.36:8080
export SWANLAB_API_KEY=Y2jpOgp5pjDMt4d9uMk3R

cd /data/home/sczc457/run/taozhang/20250817_Train_once_and_generalize

python runner.py -fromjson ../../../model/tests/final.json