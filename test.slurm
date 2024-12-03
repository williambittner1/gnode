#!/bin/bash

#SBATCH --job-name=my_python_job        # Descriptive job name
#SBATCH --nodes=1                       # Request 1 node
#SBATCH --ntasks=1                      # 1 task (usually 1 per Python script)
#SBATCH --cpus-per-task=4               # CPUs per task (adjust as needed)
#SBATCH --mem=32gb                      # Memory per node (adjust as needed)
#SBATCH --error=slurm_%A_log.err        # #SBATCH --error=slurm_%A_%a_log.err
#SBATCH --output=slurm_%A_log.out    # #SBATCH --output=slurm_%A_%a_log.out
#SBATCH --time=24:00:00                 # Time limit (HH:MM:SS)
#SBATCH --partition=gpu        # Or 'gpu' if you need GPUs
#SBATCH --gres=gpu:1                    # Uncomment and adjust if using GPUs
#SBATCH --constraint=p40              # Uncomment and specify constraints if needed
#SBATCH --requeue

# Load modules (if necessary for your environment)
# module load all/CUDA/11.8.0   # Example: load CUDA and cuDNN # Athena
module avail
module load cuda/11.7 

source ~/.bashrc

# Activate your virtual environment (crucial!)
source /users/williamb/miniconda3/bin/activate gnode_triton

# Run your Python script with full path
python /users/williamb/dev/gnode/test.py 
