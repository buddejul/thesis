#!/bin/bash
#SBATCH --account=ag_ifs_freyberger
#SBATCH --partition=intelsr_medium
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem-per-cpu=400M

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
# Memory usage was 447MB for 20 runs on a single core. Hence, 1GB or 2GB per core
# should be sufficient.

# Info on --ntasks and --cpus-per-task:
# https://stackoverflow.com/questions/51139711/hpc-cluster-select-the-number-of-cpus-and-threads-in-slurm-sbatch

# I think in the end we can just increase the cpus-per-task as desired?
# And set ntasks to 1? I think the CPUs don't need to be on a single node.
# Note: Checked ntasks=2 and cpus-per-task=1 which works fine with pytask parallel.

# 60 Reps with two tasks in parallel: 25 minutes, 402 MB Ram

# Runtime smaller runs: 60 cores-per-node, 50 sims per task, 120 tasks: 40 minutes.

# --------------------------------------------------------------------------------------
# Start script
# --------------------------------------------------------------------------------------
source ~/.bashrc
conda deactivate
conda activate thesis

# Use the dask backend for parallelization
# https://pytask-parallel.readthedocs.io/en/stable/quickstart.html
pytask --parallel-backend loky -n 60 -m hpc_pyvmte
