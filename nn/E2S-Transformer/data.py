import os
import random
import numpy as np
import pandas as pd
from typing import Tuple

import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class EEGDataset(Dataset):
    def __init__(self, config: dict):
        '''
        path: path to sections (folders)
        audio_maps: two-level map: section names -> labels -> audio_paths
        fragment_lengtht: length of fragment after label
        partition_size: number of nonzero labels in each csv file
        '''
        super().__init__()
        rnd = random.Random(config['seed'])
        
        self.sections = os.listdir(config['path'])
        rnd.shuffle(self.sections)
        assert set(self.sections) == set(config['audio_maps'].keys()), "Sections must be the same!"
        self.audio_maps = config['audio_maps']
        
        all_paths = []
        for sec in self.sections:
            l = os.listdir(os.path.join(config['path'], sec))
            rnd.shuffle(l)
            all_paths.append([os.path.join(config['path'], sec, file) for file in l])
                
        # all_paths = [[os.path.join(path, sec, file) for file in sorted(os.listdir(os.path.join(path, sec)))] for sec in self.sections]
        num_all_files = [len(elem) for elem in all_paths]
        splits = [int(elem * config['val_ratio']) for elem in num_all_files]
        
        self.val_paths = [sec_paths[:split] for sec_paths, split in zip(all_paths, splits)]
        self.paths = [sec_paths[split:] for sec_paths, split in zip(all_paths, splits)]
        
        self.sec_num_files = [len(elem) for elem in self.paths]
        self.sec_cumnum = np.cumsum(self.sec_num_files) * config['partition_size']
        self.total_num_files = sum(self.sec_num_files)
        
        self.sec_num_val_files = [len(elem) for elem in self.val_paths]
        self.sec_val_cumnum = np.cumsum(self.sec_num_val_files) * config['partition_size']
        self.total_num_val_files = sum(self.sec_num_val_files)
        
        self.partition_size = config['partition_size']
        self.fragment_length = config['fragment_length']
        self.sr = config['audio_sr']
        self.sound_channel = config['sound_channel']
        self.val_mode = False
        self.config = config
        
    def __len__(self) -> int:
        num = self.total_num_val_files if self.val_mode else self.total_num_files
        return num * self.partition_size
    
    def set_val_mode(self, mode: bool):
        '''
        Switch between train/val subsets
        '''
        assert mode in [True, False], "Incorrect mode type!"
        self.val_mode = mode
        return self
    
    def to_section(self, idx: int) -> Tuple[int, int]:
        '''
        Get file section and inner index by its absolute index
        '''
        cumnum = self.sec_val_cumnum if self.val_mode else self.sec_cumnum
        section = np.where(idx < cumnum)[0][0]
        section_idx = idx if (section == 0) else (idx - cumnum[section - 1])
        return section, section_idx
    
    def get_audio(self, section: str, label: int) -> torch.Tensor:
        '''
        Get audio by section and corresponding label
        '''
        section_name = self.sections[section]
        audio, current_sr = torchaudio.load(self.audio_maps[section_name][label])
        audio = torchaudio.functional.resample(audio, orig_freq=current_sr, new_freq=self.sr)
        return audio[self.sound_channel]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        int idx: file ID
        return: EEG fragment with its corresponding audio
        '''
        section, section_idx = self.to_section(idx)
        paths_source = self.val_paths if self.val_mode else self.paths
        file_path = paths_source[section][section_idx // self.partition_size]
        
        start = (section_idx % self.partition_size) * self.fragment_length
        end = start + self.fragment_length
        
        data = pd.read_feather(file_path).to_numpy()
        x, label = torch.tensor(data[start:end, 1:]), data[start, 0].astype(int)
        
        audio = self.get_audio(section, label)
        
        # Cut model inputs so that they match desirable sizes
        E, S = self.config['in_seq_len'], self.config['sound_size']
        x = x[:E] if x.size(0) >= E else nn.functional.pad(x, (0, E-x.size(0)), value=0)
        audio = audio[:S] if audio.size(0) >= S else nn.functional.pad(audio, (0, S-audio.size(0)), value=0)
        
        x = x.t()  # (n_channels, in_seq_len)
        
        return x.float(), audio.float()

def get_dl(config, batch_size=32):
    train_ds = EEGDataset(config=config)
    val_ds = EEGDataset(config=config).set_val_mode(True)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(train_ds)
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(val_ds)
    )
    return train_dl, val_dl