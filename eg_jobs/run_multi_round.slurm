#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=kill-shared
#SBATCH -J restrict_priors
#SBATCH --output ./%x_%j.out
#SBATCH --error ./%x_%j.err
#SBATCH -t 48:00:00
#SBATCH --ntasks=1	
#SBATCH --cpus-per-task=1
##SBATCH --exclusive
#SBATCH --mem=60G
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=runburg@hawaii.edu

module purge
module load lang/Anaconda3
source activate sbi
#export LD_LIBRARY_PATH=/usr/lib64/libstdc++.so.6
echo Installed packages
conda list
pip install sbi --user

echo Running SBI...
python multi_round_inference.py two_params.yaml 5
