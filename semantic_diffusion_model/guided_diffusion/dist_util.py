"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist
try:
    from mpi4py import MPI
except Exception:
    MPI = None


# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


"""def setup_dist():
    #Setup a distributed process group.
    
    if dist.is_initialized():
        return

    #comm = MPI.COMM_WORLD

    # ---------- Fallback: no MPI available ----------
    if MPI is None:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_find_free_port()))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        return
    # ---------- MPI available ----------
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    # If we're effectively single-process, don't init torch.distributed.
    if world_size == 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_find_free_port()))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        return
    
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")"""

def setup_dist():
    """
    Setup a distributed process group.
    Works in 3 cases:
      1) MPI available -> use MPI rank/size to set env and init process group.
      2) No MPI (Windows/local) -> init single-process group (WORLD_SIZE=1).
      3) Already initialized -> do nothing.
    """
    if dist.is_initialized():
        return

    # ---- Local / no MPI fallback ----
    if MPI is None:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_find_free_port()))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")

        dist.init_process_group(backend="gloo", init_method="env://")
        return

    # ---- MPI path ----
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    backend = "gloo" if not th.cuda.is_available() else "nccl"
    hostname = "127.0.0.1" if backend == "gloo" else socket.gethostbyname(socket.getfqdn())

    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["MASTER_PORT"] = str(comm.bcast(_find_free_port(), root=0))
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("LOCAL_RANK", "0")

    dist.init_process_group(backend=backend, init_method="env://")


"""def dev():
    
    #Get the device to use for torch.distributed.
    
    if th.cuda.is_available():
        return th.device(f"cuda:{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
    return th.device("cpu")"""

import os
import torch as th
import torch.distributed as dist

def dev():
    """
    Get the device to use for torch.distributed / local training.
    Works with/without MPI.
    """
    if not th.cuda.is_available():
        return th.device("cpu")

    # If distributed is not initialized -> single GPU local
    if not (dist.is_available() and dist.is_initialized()):
        return th.device("cuda:0")

    # Distributed initialized: use LOCAL_RANK if present, else rank%GPUS_PER_NODE
    local_rank = os.environ.get("LOCAL_RANK", None)
    if local_rank is not None:
        return th.device(f"cuda:{int(local_rank)}")

    # Fallback
    rank = dist.get_rank()
    gpus_per_node = int(os.environ.get("GPUS_PER_NODE", "1"))
    return th.device(f"cuda:{rank % gpus_per_node}")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    chunk_size = 2 ** 30  # MPI has a relatively small size limit
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        MPI.COMM_WORLD.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i: i + chunk_size])
    else:
        num_chunks = MPI.COMM_WORLD.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)

    return th.load(io.BytesIO(data), **kwargs)


"""def sync_params(params):
    #Synchronize a sequence of Tensors across ranks from rank 0.
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)"""

def sync_params(params):
    """
    Synchronize a sequence of Tensors/Parameters across ranks.
    In PyTorch 2.x, broadcast() is in-place, so do it under no_grad.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    with th.no_grad():
        for p in params:
            dist.broadcast(p.data, 0)

def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
