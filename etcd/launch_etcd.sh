#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=500M
#SBATCH --job-name=etcd-cluster
#SBATCH --nodelist=slurmd-[97-99]
#SBATCH --output=/storage/openpsi/experiments/etcd/logs/etcd.out

# Function to extract clean IP from scontrol getaddrs output
get_node_ip() {
    local node=$1
    scontrol getaddrs $node | awk -F': ' '{print $2}' | awk -F':' '{print $1}'
}

SLURM_JOB_NODELIST="slurmd-[97-99]"

# Get node IPs
NODE_IPS=()
for node in $(scontrol show hostnames $SLURM_JOB_NODELIST); do
    NODE_IPS+=($(get_node_ip $node))
done

NODE_IPS=${NODE_IPS[*]} srun --mpi=pmi2 --ntasks=3 --ntasks-per-node=1 --nodes=3 \
    --cpus-per-task=8 --mem-per-cpu=500M --nodelist=$SLURM_JOB_NODELIST \
    --export=ALL,NODE_IPS \
    singularity exec --pid --no-home --writable-tmpfs \
    --bind /storage:/storage /storage/openpsi/users/bowei.fw/sglang-sif/sglang2501-bf16.sif \
    bash /storage/openpsi/experiments/etcd/start_local.sh
