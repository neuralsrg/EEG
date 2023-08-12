import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate

# from data import get_dl
# from model import E2STransformer

cfg = OmegaConf.load('./nn/E2S-Transformer/config.yaml')
train_ds = instantiate(cfg.dataset)

print(len(train_ds))