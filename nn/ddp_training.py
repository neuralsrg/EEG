import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import os


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def test(rank: int, world_size: int):
    tensor_list = [torch.zeros(2, dtype=torch.int64).to(rank) for _ in range(2)]
    tensor = (torch.arange(2, dtype=torch.int64) + 1 + 2 * rank).to(rank)
    print(f'[{rank}] tensor = {tensor}')
    dist.all_gather(tensor_list, tensor)
    print(f'[{rank}] tensor_list = {tensor_list}')


def main(rank: int, world_size: int):
    ddp_setup(rank, world_size)
    test(rank, world_size)
    destroy_process_group()


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description='E2S Transformer DDP training job.')
    # parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    # args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    # mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
    mp.spawn(main, args=(world_size, ), nprocs=world_size)