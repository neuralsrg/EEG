import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
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
    t = torch.empty(1).fill_(rank).to(rank)
    print(f'[{rank}] t = {t}')

    if rank == 0:
        tensor_list = [torch.empty(1) for _ in range(world_size)]
        torch.distributed.gather(t, gather_list=tensor_list, dst=0)
        print(f'[{rank}] tensor_list = {tensor_list}')
        print(f'Mean value: {torch.mean(torch.tensor(tensor_list)).item()}')
    else:
        torch.distributed.gather(t, gather_list=[], dst=0)

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