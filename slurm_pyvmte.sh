#!/bin/bash
#SBATCH --account=ag_ifs_freyberger
#SBATCH --partition=intelsr_devel
#SBATCH --time=0:45:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
source ~/.bashrc
conda deactivate
conda activate thesis

# Use the dask backend for parallelization
# https://pytask-parallel.readthedocs.io/en/stable/quickstart.html
pytask --parallel-backend loky -n 2 -m hpc_pyvmte
