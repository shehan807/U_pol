#!/bin/bash
#SBATCH -A gts-jmcdaniel43-chemx
#SBATCH --job-name=dma
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH -qinferno
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=parmar@gatech.edu
#SBATCH --output=energies.%x-%j
#SBATCH --error=error.%x-%j


module load anaconda3
source activate p4env

# export myscratch=~/scratch/psi4.$PBS_JOBID
# export PSI_SCRATCH=$myscratch
# mkdir -p $PSI_SCRATCH
#
# export PSIDATADIR=""
# export PSIDATADIR=~/.conda/envs/psi4_v1.6/share/psi4/

# first, psi4 calculation
psi4 -n 8 imidazole.inp imidazole.out >& imidazole.stdout
# rm -r $PSI_SCRATCH

# now fit charges
bash fit_charges.sh
