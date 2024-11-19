#!/bin/bash
# A bash script to copy results from marvin for a specific job.
# Usage: ./copy_from_marvin.sh <job_id>
# The script will create a directory with the job_id and copy the results from marvin
# to that directory.

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <job_id>"
    exit 1
fi

JOBID=$1
USER=s93jbudd_hpc

DATE=$(date +%d-%m-%Y" "%H:%M:%S)

DIR_SOURCE="./marvin/$JOBID"

cd C:/Users/budde/projects/thesis/src/thesis/ || exit

# Create a directory with the job_id
mkdir -p $DIR_SOURCE

# Compress
tar -czf $DIR_SOURCE.tar.gz $DIR_SOURCE

echo "[$DATE] Results compressed: $JOBID"
