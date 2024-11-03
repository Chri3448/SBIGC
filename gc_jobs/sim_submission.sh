#!/bin/bash
#SBATCH --job-name=sbi
#SBATCH --partition=shared
## 3 day max run time for public partitions, except 4 hour max runtime for the sandbox partition
#SBATCH --time=0-00:30:00 ## time format is DD-HH:MM:SS


#SBATCH --cpus-per-task=1
#SBATCH --mem=20000 ## max amount of memory per node you require ##19200
##SBATCH --core-spec=0 ## Uncomment to allow jobs to request all cores on a node    

#SBATCH --error=run5-%A.err ## %A - filled with jobid
#SBATCH --output=run5-%A.out ## %A - filled with jobid

## Useful for remote notification
##SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
##SBATCH --mail-user=user@test.org

#SBATCH --array=1-1000

## All options and environment variables found on schedMD site: http://slurm.schedmd.com/sbatch.html

module purge
module load lang/Anaconda3/2023.03-1
##source /home/chri3448/.bashrc
source activate env_healpy
python /home/chri3448/EPDF_ABC/gc_jobs/jobs/run9m_poisson.py $SLURM_ARRAY_TASK_ID