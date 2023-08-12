import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate

from data import get_dl
from model import E2STransformer

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
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


if __name__ == '__main__':
    main()