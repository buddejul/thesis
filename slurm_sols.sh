#!/bin/bash
#SBATCH --account=ag_ifs_freyberger
#SBATCH --partition=intelsr_devel
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=400M

# --------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------
# Solve model using pyvmte.

# --------------------------------------------------------------------------------------
# Start script
# --------------------------------------------------------------------------------------
source ~/.bashrc
conda deactivate
conda activate thesis

# Use the dask backend for parallelization
# https://pytask-parallel.readthedocs.io/en/stable/quickstart.html
pytask --parallel-backend loky -n 5 -m pyvmte_sols
