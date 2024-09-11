import ray
# import cupy
# import ray.util.collective as col
# from cupyx.profiler import time_range
import time
import torch
import torch.distributed as dist
import socket


@ray.remote(num_gpus=1)
class Worker:
    def __init__(self, rank, world_size, warmup: int, repeat: int, n_bytes: int, dtype=torch.float16):
        self.rank = rank
        self.world_size = world_size
        self.n_bytes = n_bytes
        self.warmup = warmup
        self.nel = n_bytes // dtype.itemsize
        self.repeat = repeat
        self.dtype=dtype
        # self.buffer = torch.randn(self.nel, dtype=cupy.float32, device="cuda")

    def get_addr(self):
        return socket.gethostbyname(socket.gethostname())

    def setup(self, head):
        dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size, init_method=f"tcp://{head}:7777")

    def compute(self):
        buf = torch.randn(self.nel, dtype=self.dtype, device="cuda")
        for _ in range(5):
            dist.all_reduce(buf)
        t_total = 0
        for _ in range(self.repeat):
            torch.cuda.synchronize()
            tik = time.perf_counter()
            # with time_range('gpu_operation', color_id=0) as tr:
            dist.all_reduce(buf)
            torch.cuda.synchronize()
            t_total += time.perf_counter() - tik
        bw = self.repeat * self.n_bytes / 1e9 / (t_total)
        print(f"BW: {bw} GBps")

ray.init()
warmup = 5
repeat = 10
world_size= 40
n_bytes = 4 * 1024**3
workers = []
for i in range(world_size):
    w = Worker.remote(rank=i, world_size=world_size, warmup=warmup, repeat=repeat, n_bytes=n_bytes)
    workers.append(w)
head = ray.get(workers[0].get_addr.remote())
ray.get([w.setup.remote(head=head) for w in workers])
ray.get([w.compute.remote() for w in workers])
# init_refs.append(w.setup.remote(i, world_size))
# ray.get(init_refs)

# Invoke allreduce remotely
# ray.get([w.compute.remote() for w in workers])