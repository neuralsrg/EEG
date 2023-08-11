import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import os

from data import ToyDataset, get_dl
from model import get_model


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

def gather_test(rank: int):
    t = torch.empty(1).fill_(rank).to(rank)
    print(f'[{rank}] tensor = {t}')
    if rank == 0:
        tensor_list = [torch.empty(1).to(rank) for _ in range(2)]
        dist.gather(t, tensor_list)
        print(f'[{rank}] tensor_list = {tensor_list}')
    else:
        dist.gather(t)

def train(rank, model, train_dl, criterion, optimizer):
    model = DDP(model.to(rank), device_ids=[rank])

    for x, label in train_dl:
        print(f'{rank} Input shape: {x.size()}')
        optimizer.zero_grad()
        x, label = x.to(rank), label.to(rank)
        pred = model(x)
        loss = criterion(pred, label)
        loss.backward()
        print(f'[{rank}] Grad: {model.weight.grad}')
        optimizer.step()
    
    print(f'[{rank}] Final weight: {model.weight}')

def main(rank: int, world_size: int):
    ddp_setup(rank, world_size)
    model = get_model()
    train_dl = get_dl(ToyDataset())
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters, lr=0.01)
    train(rank=rank, model=model, train_dl=train_dl, criterion=criterion, optimizer=optimizer)
    destroy_process_group()


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description='E2S Transformer DDP training job.')
    # parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    # args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    # mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
    mp.spawn(main, args=(world_size, ), nprocs=world_size)