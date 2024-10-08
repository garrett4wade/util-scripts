import argparse
import os
import subprocess
import socket
import time
import socket
from contextlib import closing
import getpass

user = getpass.getuser()


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def copy_code_repo(args, container_name):
    first_node = f"{args.node_prefix}{args.start_idx:02d}"
    cmds = [
        f"docker exec {container_name} rm -rf /realhf",
        f"docker cp /mnt/bs_fs/{user}/distributed_llm {container_name}:/realhf",
        (f"docker exec {container_name} cp -r /_realhf/realhf/_C /realhf/realhf/"),
    ]
    for c in cmds:
        subprocess.run(f"ssh {user}@{first_node} \"{c}\"", shell=True)


def setup_ray(args):
    assert args.end_idx >= args.start_idx, (args.start_idx, args.end_idx)
    first_node = f"{args.node_prefix}{args.start_idx:02d}"
    container_name = f"{args.container_name}-{user}"
    print(f"Setting up docker container in the head node...")
    # start local docker image
    cmd = (
        f"docker run -dit --gpus all --network host --name {container_name} "
        f"{args.docker_args} {_get_default_docker_args()} {args.image_name} bash"
    )
    subprocess.run(f"ssh {user}@{first_node} \"{cmd}\"", shell=True)
    copy_code_repo(args, container_name)

    # Find head IP
    head_ip = socket.gethostbyname(first_node)
    print(f"Finish. Head IP address: {head_ip}. Starting Ray head...", flush=True)

    # start ray head
    port = int(find_free_port())
    cmd = (
        f"docker exec {container_name} ray start --head --node-ip-address={head_ip} "
        f"--port={port} --num-cpus={args.cpu} --num-gpus={args.gpu}"
    )
    subprocess.run(f"ssh {user}@{first_node} \"{cmd}\"", shell=True)

    def execute_slave_cmd(c):
        cc= f"pdsh -R ssh -w {args.node_prefix}[{args.start_idx+1:02d}-{args.end_idx:02d}] {c}"
        print(f"running `{cc}`")
        subprocess.run(cc, shell=True)

    # start ray slaves
    ray_slave_cmd = (
        f"ray start --address={head_ip}:{port} "
        f" --num-cpus={args.cpu} --num-gpus={args.gpu}"
    )
    slave_cmds = [
        (
            f"docker run -dit --gpus all --network host --name {container_name} "
            f"{args.docker_args} {_get_default_docker_args()} {args.image_name} bash"
        ),
        f"docker exec {container_name} rm -rf /realhf",
        f"docker cp /mnt/bs_fs/{user}/distributed_llm {container_name}:/realhf",
        f"docker exec {container_name} {ray_slave_cmd}",
        (f"docker exec {container_name} cp -r /_realhf/realhf/_C /realhf/realhf/"),
    ]

    if args.end_idx > args.start_idx:
        for c in slave_cmds:
            execute_slave_cmd(c)

    print("=" * 100)
    print(
        f" Ray cluster setup finishes! Container name: `{container_name}`. ".center(
            100, "="
        )
    )
    print(
        f" You can check the current status by running: `docker exec {container_name} ray status`. ".center(
            100, "="
        )
    )
    print(
        f" To shutdown the ray cluster, run: `python3 rayc.py stop`. ".center(100, "=")
    )
    print(
        f" Now you can enter the docker container to run your code: `docker exec -it {container_name} bash`. ".center(
            100, "="
        )
    )
    print("=" * 100)


def destroy_ray(args):
    container_name = f"{args.container_name}-{user}"
    cmd = f'pdsh -w {args.node_prefix}[{args.start_idx:02d}-{args.end_idx:02d}] -R ssh "docker rm -f {container_name}"'
    # for i in reversed(range(args.start_idx, args.end_idx + 1)):
    #     node = f"{args.node_prefix}{i:02d}"
    #     _cmd = f"ssh {os.getlogin()}@{node} \"docker rm -f {container_name}\""
    subprocess.run(cmd, shell=True)
    # print(f">>> Finished removing containers on node `{node}`.")


def _get_default_docker_args():
    flags = [
        f"-v /mnt/bs_fs:/mnt/bs_fs",
    ]
    flags += ["--device /dev/infiniband/rdma_cm"]
    for i in range(9):
        flags.append(f"--device /dev/infiniband/uverbs{i}")
    flags.append("--shm-size=100gb")
    flags.append("--ulimit memlock=-1")
    flags.append("--ipc=host")
    flags.append("--ulimit stack=67108864")
    return " ".join(flags)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", help="sub-command help")
    subparsers.required = True

    subparser = subparsers.add_parser("start", help="starts an experiment")
    subparser.add_argument(
        "--node_prefix",
        "-p",
        type=str,
        default="trainer",
        help="The prefix of the node. Should be 'trainer' on the JS cloud.",
    )
    subparser.add_argument(
        "--start_idx", "-s", type=int, default=1, help="The start node index, inclusive"
    )
    subparser.add_argument(
        "--end_idx",
        "-e",
        type=int,
        default=16,
        help="The end node index, inclusive. If you want to use node trainer[01-05], set -s 1 -e 5",
    )
    subparser.add_argument(
        "--container_name",
        type=str,
        default="raycluster",
        help="The container name on all nodes. Should be changed if you start multiple experiments",
    )
    subparser.add_argument(
        "--image_name",
        type=str,
        default="real-gpu:js-math",
        help="docker image name. please change it to your desired one",
    )
    subparser.add_argument(
        "--docker_args",
        type=str,
        default="",
        help="additional arguments when starting docker container. See _get_default_docker_args() for existing arguments, including mounting.",
    )
    subparser.add_argument(
        "--cpu",
        type=int,
        default=180,
        help="Number of CPUs used per node. Should < 192",
    )
    subparser.add_argument(
        "--gpu", type=int, default=8, help="Number of GPUs used per node. usually 8"
    )
    subparser.set_defaults(func=setup_ray)

    subparser = subparsers.add_parser(
        "stop", help="stops an experiment, indexed by container names"
    )
    subparser.add_argument(
        "--node_prefix",
        "-p",
        type=str,
        default="trainer",
        help="The prefix of the node. Should be 'trainer' on the JS cloud.",
    )
    subparser.add_argument(
        "--start_idx", "-s", type=int, default=1, help="The start node index, inclusive"
    )
    subparser.add_argument(
        "--end_idx",
        "-e",
        type=int,
        default=16,
        help="The end node index, inclusive. If you want to use node trainer[01-05], set -s 1 -e 5",
    )
    subparser.add_argument(
        "--container_name",
        type=str,
        default="raycluster",
        help="The container name on all nodes. Should be changed if you start multiple experiments",
    )
    subparser.set_defaults(func=destroy_ray)

    args = parser.parse_args()
    args.func(args)
