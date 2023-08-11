import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class ToyDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
    
    def __len__(self) -> int:
        return 100

    def __getitem__(self, index):
        return torch.rand(25), torch.randint(0, 2, (8,))


def get_dl(dataset, batch_size=10):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )