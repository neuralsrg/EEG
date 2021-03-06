# EEG decoding

## For anyone willing to clone / pull repository without LFS files:
```
GIT_LFS_SKIP_SMUDGE=1 git clone SERVER-REPOSITORY
git lfs pull --include "*.h5" # to pull LFS files when needed
```

### Goal:
Given `correlation matrix` / `edf file`, restore the sound spectrum.

Two different approaches: using *pseudo-inverse matrix* & *using NeuralNet*

### Datasets:
> EEG/nn/datasets

.h5 file keys: `train_set_x`, `train_set_y`, `test_set_x`, `test_set_y`

### Scripts:
* `get_file.sh` creates cURL request to download file from github (large file) storage
* `lfs_request.sh` creates cURL request using pointer-file for files stored in git-lfs (seems outdated)
