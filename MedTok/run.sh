#! /bin/bash
#SBATCH --job-name=medical_code
#SBATCH --partition=[Your_Partition_Name]
#SBATCH --account=[Your_Account_Name]
#SBATCH --mem=250G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=16
#SBATCH --time=0-72:00


source /data/yuyu/anaconda3/bin/activate MedTok
WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 1234 ../train_MedTok.py
