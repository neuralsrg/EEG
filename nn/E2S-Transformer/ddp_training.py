import os
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

from omegaconf import OmegaConf
from hydra.utils import instantiate

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from data import get_dl
from trainer import Trainer
from model import E2STransformer, NoamAnnealing


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

# @hydra.main(version_base=None, config_path=".", config_name="config")
# def get_training_data(cfg: DictConfig):
def get_training_data(cfg):
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
        conv_module_dropout=cfg.model.conv_module_dropout,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.base_lr)
    scheduler=NoamAnnealing(optimizer=optimizer, d_model=cfg.model.d_model, warmup_steps=len(train_dl) // 5,
                            min_lr=cfg.training.min_lr)
    criterion = torch.nn.MSELoss()
    return train_dl, val_dl, model, optimizer, scheduler, criterion

def main(rank: int, world_size: int):
    ddp_setup(rank, world_size)
    cfg = OmegaConf.load("config.yaml")
    train_dl, val_dl, model, optimizer, scheduler, criterion = get_training_data(cfg)
    trainer = Trainer(
        gpu_id=rank,
        train_dl=train_dl,
        val_dl=val_dl,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        n_epochs=cfg.training.n_epochs,
        batch_size=cfg.training.batch_size,
        step_every=cfg.training.step_every,
        model_checkpoint_path=cfg.training.model_checkpoint_path,
        load_from=cfg.training.load_from
    )
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description='E2S Transformer DDP training job.')
    # parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    # args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    # mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
    mp.spawn(main, args=(world_size, ), nprocs=world_size)