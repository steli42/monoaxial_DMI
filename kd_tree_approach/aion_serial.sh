#!/bin/bash -l
### Request a single task using one core on one node for 5 minutes in the batch queue
#SBATCH -N 1
#SBATCH -J skyrmion_energies
##SBATCH -dependency singleton
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --time=0-48:00:00
#SBATCH -p batch
#SBATCH --mem 218G
#SBATCH -o logs/%x-%j.out  # log goes into logs/<jobname>-<jobid>.out

print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
# Safeguard for NOT running this launcher on access/login nodes
module purge || print_error_and_exit "No 'module' command"
# List modules required for execution of the task

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

TASK=${TASK:=$(pwd)/run.sh}
srun ${TASK} $1