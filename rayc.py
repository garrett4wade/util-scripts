import argparse
import os
import subprocess
import socket
import time


def setup_ray(args):
    assert socket.gethostname() == f"{args.node_prefix}{args.start_idx:02d}", (socket.gethostname(), f"{args.node_prefix}{args.start_idx:02d}")
    container_name = f"{args.container_name}-{os.getlogin()}"
    print(f"Setting up docker container in the head node...")
    # start local docker image
    cmd = (f"docker run -dit --gpus all --network host --name {container_name} "
    f"{args.docker_args} {_get_default_docker_args()} {args.image_name} bash"
    )
    subprocess.run(cmd, shell=True)
    time.sleep(2)

    _cmd = "ifconfig bond0 | grep -oP 'inet\s+\K\d+(\.\d+){3}'"
    head_ip = subprocess.check_output(f'docker exec {container_name}  sh -c \"{_cmd}\"', shell=True).decode('ascii').strip()
    print(f"Finish. Head IP address: {head_ip}. Starting Ray head...")

    # start ray head
    cmd = (f"docker exec {container_name} ray start --head --node-ip-address={head_ip} "
    f"--port={args.port} --num-cpus={args.cpu} --num-gpus={args.gpu}") 
    subprocess.run(cmd, shell=True)
    print("Done. Setting up slave nodes...")

    # start ray slaves
    ray_slave_cmd = (f"ray start --address={head_ip}:{args.port} "
    f" --num-cpus={args.cpu} --num-gpus={args.gpu}")
    slave_cmd1 = (f"docker run -dit --gpus all --network host --name {container_name} "
    f"{args.docker_args} {_get_default_docker_args()} {args.image_name} bash"
    )
    slave_cmd2 = f"docker exec {container_name} {ray_slave_cmd}"

    for i in range(args.start_idx + 1, args.end_idx + 1):
        node = f"{args.node_prefix}{i:02d}"
        _cmd = f"ssh {os.getlogin()}@{node} \"{slave_cmd1}\""
        subprocess.run(_cmd, shell=True)
        _cmd = f"ssh {os.getlogin()}@{node} \"{slave_cmd2}\""
        subprocess.run(_cmd, shell=True)
        print(f"running {_cmd}")
        print(f"Node `{node}` setup finish.")
    
    print("="*100)
    print(f" Ray cluster setup finishes! Container name: `{container_name}`. ".center(100, "="))
    print(f" You can check the current status by running: `docker exec {container_name} ray status`. ".center(100, "="))
    print(f" To shutdown the ray cluster, run: `python3 rayc.py stop`. ".center(100, "="))
    print(f" Now you can enter the docker container to run your code: `docker exec -it {container_name} bash`. ".center(100, "="))
    print("="*100)



def destroy_ray(args):
    container_name = f"{args.container_name}-{os.getlogin()}"
    for i in reversed(range(args.start_idx, args.end_idx + 1)):
        node = f"{args.node_prefix}{i:02d}"
        _cmd = f"ssh {os.getlogin()}@{node} \"docker rm -f {container_name}\""
        subprocess.run(_cmd, shell=True)
        print(f">>> Finished removing containers on node `{node}`.")

def _get_default_docker_args():
    flags = [f"-v /mnt/bs_fs:/mnt/bs_fs", f"-v /mnt/bs_fs/{os.getlogin()}/distributed_llm:/realhf"]
    flags += ["--device /dev/infiniband/rdma_cm"]
    for i in range(9):
        flags.append(f"--device /dev/infiniband/uverbs{i}")
    flags.append('--shm-size=100gb')
    flags.append('--ulimit memlock=-1:-1')
    return ' '.join(flags)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", help="sub-command help")
    subparsers.required = True

    subparser = subparsers.add_parser("start", help="starts an experiment")
    subparser.add_argument("--node_prefix", "-p", type=str, default="trainer", help="The prefix of the node. Should be 'trainer' on the JS cloud.")
    subparser.add_argument("--start_idx", "-s", type=int, default=1, help="The start node index, inclusive")
    subparser.add_argument("--end_idx", "-e", type=int, default=5, help="The end node index, inclusive. If you want to use node trainer[01-05], set -s 1 -e 5")
    subparser.add_argument("--container_name", type=str, default="raycluster", help="The container name on all nodes. Should be changed if you start multiple experiments")
    subparser.add_argument("--image_name", type=str, default="real-gpu:js", help="docker image name. please change it to your desired one")
    subparser.add_argument("--docker_args", type=str, default="", help="additional arguments when starting docker container. See _get_default_docker_args() for existing arguments, including mounting.")
    subparser.add_argument("--cpu", type=int, default=180, help="Number of CPUs used per node. Should < 192")
    subparser.add_argument("--gpu", type=int, default=8, help="Number of GPUs used per node. usually 8")
    subparser.add_argument("--port", type=int, default=6379, help="Ray cluster port. Ports should be different if you start multiple ray clusters on the same node.")
    subparser.set_defaults(func=setup_ray)

    subparser = subparsers.add_parser("stop", help="stops an experiment, indexed by container names")
    subparser.add_argument("--node_prefix", "-p", type=str, default="trainer", help="The prefix of the node. Should be 'trainer' on the JS cloud.")
    subparser.add_argument("--start_idx", "-s", type=int, default=1, help="The start node index, inclusive")
    subparser.add_argument("--end_idx", "-e", type=int, default=5, help="The end node index, inclusive. If you want to use node trainer[01-05], set -s 1 -e 5")
    subparser.add_argument("--container_name", type=str, default="raycluster", help="The container name on all nodes. Should be changed if you start multiple experiments")
    subparser.set_defaults(func=destroy_ray)

    args = parser.parse_args()
    args.func(args)

