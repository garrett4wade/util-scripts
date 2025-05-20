import os

import etcd3

# 连接 etcd 集群
etcd = etcd3.client(host="10.11.18.212", port=2379)


# 测试写入
etcd.put("/slurm/test_key", "hello from slurm!")

# 测试读取
value, _ = etcd.get("/slurm/test_key")
print(f"从 etcd 读取的值: {value.decode('utf-8')}")
