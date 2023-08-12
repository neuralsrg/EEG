import os
from tqdm import tqdm, trange

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

from data import get_dl
from model import E2STransformer


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

@hydra.main(version_base=None, config_path=".", config_name="config")
def get_training_data(cfg: DictConfig):
    print('enter get data')
    # data
    train_ds = instantiate(cfg.dataset)
    val_ds = instantiate(cfg.dataset).set_val_mode(True)
    train_dl, val_dl = get_dl(train_ds=train_ds, val_ds= val_ds, batch_size=cfg.training.batch_size)

    # model 
    model = E2STransformer(
        n_channels=cfg.model.n_channels,
        n_wvt_bins=cfg.model.n_wvt_bins,
        d_model=cfg.model.d_model,
        kernel_size=cfg.model.kernel_size,
        emb_dropout=cfg.model.emb_dropout,
        in_seq_len=cfg.model.in_seq_len,
        n_fft=cfg.model.n_fft,
        hop_size=cfg.model.hop_size,
        nhead=cfg.model.nhead,
        num_encoder_layers=cfg.model.num_encoder_layers,
        num_decoder_layers=cfg.model.num_decoder_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        activation=cfg.model.activation,
        audio_sr=cfg.model.audio_sr,
        audio_paths=cfg.model.audio_paths,
        eeg_sr=cfg.model.eeg_sr,
        dj=cfg.model.dj,
        example_input=train_ds[0][0]
    )

    print(model)
    return train_dl, val_dl, model

def train(rank, model, train_dl, criterion, optimizer):

    def run_batch(x, label):
        optimizer.zero_grad()
        pred = model(x)[:, 0, :]
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        return loss.item()

    model = DDP(model.to(rank), device_ids=[rank])
    master_process = rank == 0

    for eeg, audio in (pbar := tqdm(train_dl, total=len(train_dl), disable=(not master_process))):
        loss = run_batch(eeg.to(rank), audio.to(rank))
        pbar.set_description(f'Train Loss: {loss}')
    
def validate(rank, model, criterion, val_dl):
    def run_batch(x, label):
        pred = model(x)[:, 0, :]
        loss = criterion(pred, label)
        return loss

    model = DDP(model.to(rank), device_ids=[rank]).eval()
    master_process = rank == 0

    for eeg, audio in (val_pbar := tqdm(val_dl, total=len(val_dl), disable=(not master_process))):
        loss = run_batch(eeg.to(rank), audio.to(rank))
        if master_process:
            tensor_list = [loss.new_empty(()) for _ in range(2)]
            dist.gather(loss, tensor_list)
            val_pbar.set_description(f'Mean Val Loss: {torch.tensor(tensor_list).mean().item()}')
        else:
            dist.gather(loss)
    model.train()

def main(rank: int, world_size: int):
    print('enter main')
    ddp_setup(rank, world_size)
    train_dl, val_dl, model = get_training_data()
    destroy_process_group()


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description='E2S Transformer DDP training job.')
    # parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    # args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    # mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
    mp.spawn(main, args=(world_size, ), nprocs=world_size)