#!/bin/bash
#SBATCH --account=VB-CH5HPC-002
#SBATCH --job-name=dlp
#SBATCH --output=dlp.%j.o
#SBATCH --error=dlp.%j.e
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --partition=spot-fsv2-16
#SBATCH --qos=spot-fsv2-16
#SBATCH --time=48:00:00


#Identify node correctly
source /apps/build/easy_build/scripts/id_instance.sh
source /apps/build/easy_build/scripts/setup_modules.sh

# Purge modules and load required modules
module purge
module load DL_POLY_Classic/1.10-foss-2020b

export UCX_TLS=self,tcp,sm
mpirun --oversubscribe -np 16 --host $(hostname):16  DLPOLY.X
