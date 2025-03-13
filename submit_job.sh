#!/bin/bash
#SBATCH --job-name=huutien       # Job name
#SBATCH --output=w_annatomask_res.txt      # Output file
#SBATCH --error=w_annatomask_error.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=4G                 # Memory per node (4 GB)
#SBATCH --gpus=1                 # Number of GPUs per node

cd /home/user10/huutien/simplified_anatomask
python main.py