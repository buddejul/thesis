#!/bin/bash
#SBATCH --account=ag_ifs_freyberger
#SBATCH --partition=intelsr_debug
#SBATCH --time=0:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
source ~/.bashrc
conda deactivate
conda activate thesis

# Use the dask backend for parallelization
# https://pytask-parallel.readthedocs.io/en/stable/quickstart.html
pytask --parallel-backend loky -n 1 -m hpc_pyvmte
