#!/bin/bash

# NODE_IPS is passed in by the sbatch script
IFS=' ' read -ra NODE_IPS <<< "${NODE_IPS[*]}"
NODE_ID=$SLURM_PROCID
CURRENT_IP=${NODE_IPS[$NODE_ID]}

echo "ALL node IPs: [$NODE_IPS], my node id: $SLURM_PROCID"

# ETCD configuration
ETCD_NAME="etcd-$NODE_ID"
DATA_DIR="/var/lib/etcd/etcd-$NODE_ID"  # Change to persistent storage in production
CLIENT_URL="http://${CURRENT_IP}:2379"
PEER_URL="http://${CURRENT_IP}:2380"

# Build initial cluster string
INITIAL_CLUSTER=()
for i in ${!NODE_IPS[@]}; do
    INITIAL_CLUSTER+=("etcd-$i=http://${NODE_IPS[$i]}:2380")
done
INITIAL_CLUSTER_STR=$(IFS=,; echo "${INITIAL_CLUSTER[*]}")

# Create data directory
rm -rf $DATA_DIR
mkdir -p $DATA_DIR
chmod 700 $DATA_DIR

# Write endpoints to shared location (first node only)
if [ $NODE_ID -eq 0 ]; then
    ENDPOINTS_STR=$(IFS=,; echo "${NODE_IPS[@]/%/:2379}")
    echo $ENDPOINTS_STR > /storage/openpsi/experiments/etcd/endpoints.txt
fi

# 启动 etcd (这里假设 etcd 已在 PATH 中)
/storage/openpsi/experiments/etcd/etcd-v3.5.21-linux-amd64/etcd \
    --name $ETCD_NAME \
    --data-dir $DATA_DIR \
    --listen-client-urls "$CLIENT_URL,http://127.0.0.1:2379" \
    --advertise-client-urls "$CLIENT_URL" \
    --listen-peer-urls "$PEER_URL" \
    --initial-advertise-peer-urls "$PEER_URL" \
    --initial-cluster "$INITIAL_CLUSTER_STR" \
    --initial-cluster-token "slurm-etcd-cluster" \
    --initial-cluster-state new
