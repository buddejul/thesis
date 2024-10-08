#!/bin/bash
#SBATCH --account=ag_ifs_freyberger
#SBATCH --partition=intelsr_devel
#SBATCH --time=0:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

# --------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------
# Pyvmte simulations take roughly 10 minutes (at most) for 20 simulation runs.
# Hence, on a single core a simulation
# - with 1_000 runs takes roughly 500 minutes or 8 hours and 20 minutes;
# - with 2_000 runs takes roughly 1_000 minutes or 16 hours and 40 minutes.
#
# Based on the solution regions, at least 1/3 of runs should exit early, so these are
# upper bounds.
#

# --------------------------------------------------------------------------------------
# Start script
# --------------------------------------------------------------------------------------
source ~/.bashrc
conda deactivate
conda activate thesis

# Use the dask backend for parallelization
# https://pytask-parallel.readthedocs.io/en/stable/quickstart.html
pytask --parallel-backend loky -n 2 -m hpc_pyvmte
