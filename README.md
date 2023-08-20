# EEG processing models

## E2S Transformer
* Architecture

![Alt text](https://svgur.com/i/wiH.svg)

* [Kaggle notebook](https://www.kaggle.com/code/neuralsrg/e2s-transformer)
* [Kaggle dataset](https://www.kaggle.com/datasets/mrgeodezik/internal-speech-recognition) (Private)

### Torch Distributed Data Parallel script
1. Install config handling libraries
```
pip install hydra-core --upgrade
pip install omegaconf
```
2. Install PyTorchWavelets (cuda implementation)
```
git clone https://github.com/neuralsrg/PyTorchWavelets.git
```
3. Get [DDP scripts](https://github.com/neuralsrg/EEG/tree/main/nn/E2S-Transformer/DDP)
```
wget https://raw.githubusercontent.com/neuralsrg/EEG/main/nn/E2S-Transformer/config.yaml -O config.yaml
wget https://raw.githubusercontent.com/neuralsrg/EEG/main/nn/E2S-Transformer/data.py -O data.py
wget https://raw.githubusercontent.com/neuralsrg/EEG/main/nn/E2S-Transformer/ddp_training.py -O ddp_training.py
wget https://raw.githubusercontent.com/neuralsrg/EEG/main/nn/E2S-Transformer/model.py -O model.py
wget https://raw.githubusercontent.com/neuralsrg/EEG/main/nn/E2S-Transformer/trainer.py -O trainer.py
```
4. Run DDP training
```
python ddp_training.py
```

## Dealing with LFS files (not included currently)
### For anyone willing to clone / pull repository without LFS files:
```
GIT_LFS_SKIP_SMUDGE=1 git clone SERVER-REPOSITORY
git lfs pull --include "*.h5" # to pull LFS files when needed
```

### Scripts:
* `get_file.sh` creates cURL request to download file from github (large file) storage
* `lfs_request.sh` creates cURL request using pointer-file for files stored in git-lfs (seems outdated)
