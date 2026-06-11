"""
Helpers for distributed training.
"""

import io
import os
import socket
import tempfile

import blobfile as bf
import functools
# import os
# import subprocess
# import torch
# import torch.distributed as dist
import torch.multiprocessing as mp
# from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return
    # Allow single-process launch without torchrun by setting defaults.
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        # Pick an available port to avoid collisions.
        os.environ["MASTER_PORT"] = str(_find_free_port())

    rank = int(os.environ["RANK"])
    num_gpus = th.cuda.device_count()
    # os.environ["CUDA_VISIBLE_DEVICES"] = f"{rank % num_gpus}"
    backend = "nccl"
    if os.name == "nt" or not th.cuda.is_available() or num_gpus == 0:
        backend = "gloo"

    init_method = "env://"
    if os.name == "nt":
        # Avoid TCPStore issues on Windows by using file store.
        init_file = os.path.join(tempfile.gettempdir(), "ddpm_dist_init")
        init_method = f"file:///{init_file.replace(os.sep, '/')}"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())

    rank = int(os.environ["RANK"])
    num_gpus = th.cuda.device_count()
    if th.cuda.is_available() and num_gpus > 0:
        th.cuda.set_device(rank % num_gpus)
    if init_method == "env://":
        dist.init_process_group(backend=backend, init_method=init_method)
    else:
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
        )
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"  # set to DETAIL for runtime logging.

def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """

    return th.load(path, **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
