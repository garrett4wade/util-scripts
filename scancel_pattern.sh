#!/bin/bash

# Get all job IDs and names, then filter for names starting with "singularity"
job_info=$(squeue -o "%i %j" -h | grep -E '^[0-9]+ singularity')

# Extract just the job IDs
job_ids=$(echo "$job_info" | awk '{print $1}')

# Check if any jobs were found
if [ -z "$job_ids" ]; then
    echo "No Slurm jobs found with names starting with 'singularity'"
    exit 0
fi

# Display and cancel all matching jobs
echo "Found the following jobs to cancel:"
echo "$job_info"
echo
echo "Cancelling these jobs..."
scancel $job_ids
