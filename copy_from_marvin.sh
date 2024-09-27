#!/bin/bash
# A bash script to copy results from marvin for a specific job.
# Usage: ./copy_from_marvin.sh <job_id> <marvin_username> <marvin_password>
# Example: ./copy_from_marvin.sh 12345678 my_username my_password
# The script will create a directory with the job_id and copy the results from marvin to that directory.

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <job_id> <marvin_username>"
    exit 1
fi

JOBID=$1
USER=$2

DATE=$(date +%d-%m-%Y" "%H:%M:%S)

DIR_TARGET="./marvin/job_$JOBID"

cd C:/Users/budde/projects/thesis || exit

# Create a directory with the job_id
mkdir -p $DIR_TARGET

# Copy the results from marvin
scp -r $USER@marvin.hpc.uni-bonn.de:/lustre/scratch/data/$USER-thesis/thesis/bld/ $DIR_TARGET
