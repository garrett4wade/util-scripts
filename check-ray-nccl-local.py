import argparse
import multiprocessing as mp
import os
import time

import torch
import torch.distributed as dist
from torch.profiler import profile


def _comm_impl(n_elements, dtype, rank, ranks, comm_type, group):
    if comm_type in ["scatter", "allgather", "allgather_t"]:
        t = torch.randn(n_elements // len(ranks), dtype=dtype, device="cuda")
    elif comm_type == "alltoall":
        t = torch.randn(n_elements // len(ranks), dtype=dtype, device="cuda")
    else:
        t = torch.randn(n_elements, dtype=dtype, device="cuda")
    if comm_type == "scatter":
        scatter_lis = [torch.rand_like(t) for _ in ranks]
    if comm_type == "allgather":
        allgather_lis = [torch.rand_like(t) for _ in ranks]
    if comm_type == "allgather_t":
        dst = torch.empty(
            (len(ranks), n_elements // len(ranks)), dtype=dtype, device="cuda"
        )
    if comm_type == "alltoall":
        output_list = [torch.rand_like(t) for _ in ranks]
        input_list = [torch.rand_like(t) for _ in ranks]
    torch.cuda.synchronize()
    tik = time.perf_counter_ns()
    if comm_type == "scatter":
        dist.scatter(
            t, scatter_lis if rank == ranks[0] else None, src=ranks[0], group=group
        )
    elif comm_type == "allgather":
        dist.all_gather(allgather_lis, t, group=group)
    elif comm_type == "allgather_t":
        dist.all_gather_into_tensor(dst, t, group=group)
    elif comm_type == "bcast":
        dist.broadcast(t, src=ranks[0], group=group)
    elif comm_type == "allreduce":
        dist.all_reduce(t, group=group)
    elif comm_type == "alltoall":
        dist.all_to_all(output_list, input_list, group=group)
    elif comm_type == "sendrecv":
        assert len(ranks) == 2
        if rank == ranks[0]:
            dist.send(t, dst=ranks[1], group=group)
        elif rank == ranks[1]:
            dist.recv(t, src=ranks[0], group=group)
    else:
        raise NotImplementedError()
    torch.cuda.synchronize()
    return time.perf_counter_ns() - tik


def main(rank, world_size, head_ip):
    # os.environ['NCCL_IB_HCA'] = "=mlx5_1"
    # os.environ["NCCL_NET"] = "IB"
    os.environ["NCCL_SOCKET_IFNAME"] = "bond0"
    os.environ["NCCL_IB_GID_INDEX"] = "3"
    # os.environ["NCCL_DEBUG"] = "INFO"
    # os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
    # os.environ["NCCL_IB_DISABLE"] = "0"
    # os.environ["NCCL_IB_CUDA_SUPPORT"] = "1"
    # os.environ["NCCL_ALGO"] = "RING"
    # os.environ["NCCL_NET_GDR_LEVEL"] = "SYS"
    # os.environ["NCCL_NET_GDR_READ"] = "1"
    # os.environ["NCCL_SHM_DISABLE"] = str(1)
    # os.environ["NCCL_P2P_DISABLE"] = str(1)
    torch.cuda.set_device(rank % 8)
    dist.init_process_group(
        "nccl", init_method=f"tcp://{head_ip}:7777", rank=rank, world_size=world_size
    )
    ranks = list(range(world_size))
    n_iters = 20
    g = dist.new_group(ranks=ranks, backend="nccl")
    data_size_gb = 10
    dtype = torch.half
    n_elements = data_size_gb * 1024**3 // dtype.itemsize

    for comm_type in [
        # "scatter",
        # "allgather",
        # "allgather_t",
        "allreduce",
        "bcast",
        # "alltoall",
    ]:
        if rank in ranks:
            _comm_impl(n_elements, dtype, rank, ranks, comm_type, g)

            total_t = 0
            for _ in range(n_iters):
                total_t += _comm_impl(n_elements, dtype, rank, ranks, comm_type, g)

            total_t = torch.tensor(total_t, device="cuda", dtype=torch.float32)
            dist.all_reduce(total_t, group=g)
            total_t /= len(ranks) * n_iters
            if rank == ranks[0]:
                print(
                    f"{comm_type} data {data_size_gb} GB in avg {total_t.item() / 1e3:.2f}us, "
                    f"BW {data_size_gb * 1024**3 / 1e9 / (total_t.item() / 1e9):.2f} GB/s"
                )
    dist.barrier()
    dist.destroy_process_group(g)
    dist.destroy_process_group()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrank", "-i", type=int, required=True, help="current node rank")
    parser.add_argument("--n_nodes", "-n", type=int, required=True)
    parser.add_argument("--head", type=str, required=True)
    args = parser.parse_args()
    procs = []
    for i in range(8):
        p = mp.Process(target=main, args=(i + args.nrank * 8, args.n_nodes * 8, args.head))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()