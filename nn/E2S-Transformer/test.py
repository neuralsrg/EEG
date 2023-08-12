import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    dataset = instantiate(cfg.dataset)

    print(dataset[0])

if __name__ == '__main__':
    main()