#!/bin/bash
#PBS -q normalbw 
#PBS -l ncpus=25
#PBS -l mem=128GB
#PBS -l jobfs=400GB
#PBS -l walltime=10:00:00
#PBS -l software=python
#PBS -l wd
 
# Load modules.
module load Python/3.7.3
# module load python/3.7.3-matplotlib
# module load python/3.7.3-numpy
 
# Run Python applications
python PIMC_exchange.py > $PBS_JOBID.log