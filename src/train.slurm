#!/bin/bash

#SBATCH --job-name=my_python_job        # Descriptive job name
#SBATCH --nodes=1                       # Request 1 node
#SBATCH --ntasks=1                      # 1 task (usually 1 per Python script)
#SBATCH --cpus-per-task=4               # CPUs per task (adjust as needed)
#SBATCH --mem=64gb                      # Memory per node (adjust as needed)
#SBATCH --error=/users/williamb/dev/gnode/slurm/slurm_%A.err     #SBATCH --error=slurm_%A_%a.err
#SBATCH --output=/users/williamb/dev/gnode/slurm/slurm_%A.out    #SBATCH --output=slurm_%A_%a.out
#SBATCH --time=36:00:00                 # Time limit (HH:MM:SS)
#SBATCH --partition=gpu        # Or 'gpu' if you need GPUs
#SBATCH --gres=gpu:1                    # Uncomment and adjust if using GPUs
#SBATCH --constraint=a6000              # Uncomment and specify constraints if needed
#SBATCH --requeue

# Load modules (if necessary for your environment)
# module load all/CUDA/11.8.0   # Example: load CUDA and cuDNN # Athena
# module avail
module load all/CUDA/11.7.0

source ~/.bashrc

# Activate your virtual environment (crucial!)
source /users/williamb/miniconda3/bin/activate gaussian_splatting_athena

# Run your Python script with full path
python /users/williamb/dev/gnode/src/train.py 
