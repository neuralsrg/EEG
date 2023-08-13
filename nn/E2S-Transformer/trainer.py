import os
import pickle
import numpy as np
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP


class Trainer:
    def __init__(
        self,
        gpu_id: int,
        train_dl: DataLoader,
        val_dl: DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        scaler: torch.cuda.amp.GradScaler,
        criterion: nn.Module,
        n_epochs: int,
        batch_size: int,
        step_every: int,
        model_checkpoint_path: str,
        load_from: str,
    ) -> None:
        self.gpu_id = gpu_id
        self.master_process = self.gpu_id == 0

        # dataloaders
        self.train_dl = train_dl
        self.val_dl = val_dl

        # model
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler

        if os.path.exists(load_from):
            self._load_state(load_from)
        self.criterion = criterion

        self.model = DDP(self.model.to(gpu_id), device_ids=[gpu_id])

        # training
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.step_every = step_every
        self.model_checkpoint_path = model_checkpoint_path
        self.best_val_loss = float('inf')
        self.cur_val_loss = float('inf')
        self.hist = []

    def train(self):
        def run_batch(eeg, audio, step):
            pred_encoding, encoding = self.model(eeg, audio)
            loss = self.criterion(pred_encoding, encoding) / self.step_every
            loss.backward()

            if step % self.step_every == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            return loss.item() * self.step_every

        # for epoch in trange(self.n_epochs, disable=(not self.master_process)):
        for epoch in range(self.n_epochs):
            total_batches = len(self.train_dl)

            for i, (eeg, audio) in enumerate(pbar := tqdm(self.train_dl, total=total_batches, disable=(not self.master_process), position=0, leave=True), ncols=1000):
                loss = run_batch(eeg.to(self.gpu_id), audio.to(self.gpu_id), step=i+1)
                pbar.set_description(f'T|loss:{loss:.2f}|best val:{self.best_val_loss:.2f}|cur val:{self.cur_val_loss:.2f}')

                ##############
                if i == 5:
                    break
                ##############
                if self.master_process:
                    self.hist.append((loss, 'train'))

                # if (i+1 == total_batches//2) or (i+1 == total_batches):
                if i == 1:
                    self.validate(pbar, loss)  # if tqdm OK?

            if self.master_process:
                print(f'\nEpoch {epoch} finished with the best validation loss {self.best_val_loss:.3f}.\n')
        if self.master_process:
            self._save_final_state()

    def validate(self, pbar: tqdm, train_loss: float):
        def run_batch(eeg, audio):
            pred_encoding, encoding = self.model(eeg, audio)
            loss = self.criterion(pred_encoding, encoding)
            return loss

        self.model.eval()
        if self.master_process:
            losses = []  # losses across all validation data
        with torch.no_grad():
            total_batches = len(self.val_dl)
            for i, (eeg, audio) in enumerate(self.val_dl):
                loss = run_batch(eeg.to(self.gpu_id), audio.to(self.gpu_id))
                if self.master_process:
                    tensor_list = [loss.new_empty(()) for _ in range(2)]
                    dist.gather(loss, tensor_list)
                    mean_val_loss = torch.tensor(tensor_list).mean().item()
                    losses.append(mean_val_loss)
                    self.hist.append((mean_val_loss, 'val'))
                    pbar.set_description(f'Val|loss:{train_loss:.2f}|best val:{self.best_val_loss:.2f}|cur val:{mean_val_loss:.2f}')
                else:
                    dist.gather(loss)

                ##############
                if i == 5:
                    break
                ##############
        
        if self.master_process:
            self.cur_val_loss = np.mean(losses)
            if self.cur_val_loss < self.best_val_loss:
                self.best_val_loss = self.cur_val_loss
                self._save_checkpoint('best_model.pt')

        self.model.train()

    def _save_checkpoint(self, name: str):
        ckp = self.model.module.state_dict()
        if not os.path.exists(self.model_checkpoint_path):
            os.makedirs(self.model_checkpoint_path)
        PATH = os.path.join(self.model_checkpoint_path, name)
        torch.save(ckp, PATH)
        # print(f'Best validation loss achieved: {self.best_val_loss:.3f}. Model checkpoint saved as {PATH}.')
    
    def _save_final_state(self):
        PATH = os.path.join(self.model_checkpoint_path, 'final_state')
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        names = ['model', 'optimizer', 'scheduler', 'scaler']
        entities = [self.model.module, self.optimizer, self.scheduler, self.scaler]

        for name, entity in zip(names, entities):
            torch.save(entity.state_dict(), os.path.join(PATH, f'{name}.pt'))
        
        with open(os.path.join(PATH, 'hist.pickle'), 'wb') as handle:
            pickle.dump(self.hist, handle)
        print(f'Saved final state at {PATH}')

    def _load_state(self, PATH):
        names = ['model', 'optimizer', 'scheduler', 'scaler']
        entities = [self.model, self.optimizer, self.scheduler, self.scaler]

        for name, entity in zip(names, entities):
            entity.load_state_dict(torch.load(os.path.join(PATH, f'{name}.pt')))
        
        print('Successfully loaded pretrained state dict.')