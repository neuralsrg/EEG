import os
import random
import numpy as np
import pandas as pd
from typing import Tuple, Iterable

import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class EEGDataset(Dataset):
    path: str
    audio_maps: dict
    fragment_length: int
    partition_size: int
    sample_rate: int
    sound_channel: int
    val_ratio: float
    seed: int

    def __init__(
        self,
        path: str,
        audio_maps: dict,
        fragment_length: int,
        partition_size: int,
        sample_rate: int,
        sound_channel: int,
        val_ratio: float,
        sound_size: int,
        in_seq_len: int,
        seed: int,
    ) -> None:
        '''
        path: path to sections (folders)
        audio_maps: two-level map: section names -> labels -> audio_paths
        fragment_lengtht: length of fragment after label
        partition_size: number of nonzero labels in each csv file
        '''
        super().__init__()
        self.path = path
        self.audio_maps = audio_maps
        self.fragment_length = fragment_length
        self.partition_size = partition_size
        self.sample_rate = sample_rate
        self.sound_channel = sound_channel
        self.val_ratio = val_ratio
        self.sound_size = sound_size
        self.in_seq_len = in_seq_len
        self.seed = seed

        self.transforms = None

        rnd = random.Random(seed)
        
        self.sections = os.listdir(path)
        rnd.shuffle(self.sections)
        assert set(self.sections) == set(audio_maps.keys()), "Sections must be the same!"
        self.audio_maps = audio_maps
        
        all_paths = []
        for sec in self.sections:
            l = os.listdir(os.path.join(path, sec))
            rnd.shuffle(l)
            all_paths.append([os.path.join(path, sec, file) for file in l])
                
        # all_paths = [[os.path.join(path, sec, file) for file in sorted(os.listdir(os.path.join(path, sec)))] for sec in self.sections]
        num_all_files = [len(elem) for elem in all_paths]
        splits = [int(elem * val_ratio) for elem in num_all_files]
        
        self.val_paths = [sec_paths[:split] for sec_paths, split in zip(all_paths, splits)]
        self.paths = [sec_paths[split:] for sec_paths, split in zip(all_paths, splits)]
        
        self.sec_num_files = [len(elem) for elem in self.paths]
        self.sec_cumnum = np.cumsum(self.sec_num_files) * partition_size
        self.total_num_files = sum(self.sec_num_files)
        
        self.sec_num_val_files = [len(elem) for elem in self.val_paths]
        self.sec_val_cumnum = np.cumsum(self.sec_num_val_files) * partition_size
        self.total_num_val_files = sum(self.sec_num_val_files)
        
        self.val_mode = False
        
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
        audio = torchaudio.functional.resample(audio, orig_freq=current_sr, new_freq=self.sample_rate)
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
        x, label = torch.tensor(data[start:end, 1:]), int(data[start, 0])
        
        audio = self.get_audio(section, label)
        
        # Cut model inputs so that they match desirable sizes
        E, S = self.in_seq_len, self.sound_size
        x = x[:E] if x.size(0) >= E else nn.functional.pad(x, (0, E-x.size(0)), value=0)
        audio = audio[:S] if audio.size(0) >= S else nn.functional.pad(audio, (0, S-audio.size(0)), value=0)
        
        x = x.t()  # (n_channels, in_seq_len)
        x, audio = x.float(), audio.float()
        
        if self.transforms is not None:
            for t in self.transforms:
                x, audio = t((x, audio))
        
        return x, audio

def get_dl(train_ds: Dataset, val_ds: Dataset, batch_size: int = 32):
    dls = [DataLoader(ds, batch_size=batch_size, pin_memory=True,
                      shuffle=False, sampler=DistributedSampler(ds))
           for ds in (train_ds, val_ds)]
    return dls


class AudioAugment(object):
    def __init__(self, sigma: int):
        self.sigma = sigma

    def __call__(self, item: Iterable):
        eeg, audio = item
        return eeg, torch.FloatTensor(*audio.size()).to(audio.device).normal_(0, self.sigma)