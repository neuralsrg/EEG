# Datasets for EDF neural nets 

### DSGenerator (Dataset Generator) tutorial 

Creates datasets based on .EDF file(s).

1. Create `generator`
```
path = '/path/to/file.EDF'
generator = WindowGenerator([path1]) # list here is important
```
You can also concatenate **multiple** .EDF files into one dataset:
```
path1 = '/path/to/file.EDF'
path2 = '/path/to/file.EDF'
generator = WindowGenerator([path1, path2])
```

2. (Optional) Learn some dataset info (number of events, number of channels, filtering info, `df.describe` etc.)
```
generator.eeg_info()
```
Each electrode channel is `normalized`. You can get `normalization vector`:
```
generator.normalize
```

3. If you intend to use `noise` along with `listen` and `repeat` data, you **should** specify noise indices (one index per noise interval -- first index in each interval). I have already computed correct indices (for 300ms time lapse), load them from `bash_and_bond_noise.json` and `bash_noise.json`:
```
import os
import numpy as np
import codecs, json 

path = "/content/data"

obj_text = codecs.open(os.path.join(path, 'bash_and_bond_noise.json'), 'r', encoding='utf-8').read()
loaded = json.loads(obj_text)
bash_bond = np.array(loaded)

obj_text = codecs.open(os.path.join(path, 'bash_noise.json'), 'r', encoding='utf-8').read()
loaded = json.loads(obj_text)
bash = np.array(loaded)
```

Pass these indices to `generator.create_dataset(indices_noise=bash)` in case of single EDF dataset (based on `bash_phoneme.EDF`) and `generator.create_dataset(indices_noise=bash_bond)` in case of dataset based on both `bash_phoneme.EDF` and `bond_phoneme.EDF` 

4. Now you can create dataset with `generator.create_dataset()`
```
generator.create_dataset(
	 event_length : Optional[int] = 300, # time lapse after each label to include in dataset (in ms since frequency ~ 1000Hz)
	 indices_noise : Optional[np.array] = None, # loaded / created indices for noise data (first indices in each interval)
	 plot_indices : Optional[bool] = True, # whether to plot indices to check if they overlap
	 skip_in_repeat : Optional[int] = 100, # shift each repeat interval to the right by skip_in_repeat features
	 listen_repeat_noise : Optional[Tuple[bool, bool, bool]] = [True, False, True], # which data to include in dataset
	 train_val_test : Optional[Tuple[float, float, float]] = [.8, .2, 0], # train / val / test ratios
	 shuffle_before_splitting (optional) -- bool. Whether to shuffler data before splitting into
    	train / val / test or after
	 batch_size : Optional[int] = 32, # batch size
	 split_windows : Optional[bool] = False, # whether to split time intervals into smaller windows
	 channels : Optional[Sequence[int]] = [], # which channels to use (index array)
	 window_size : Optional[int] = 128, # if split_windows, window size
	 shift : Optional[int] = None, # if split_windows, hop size between each window (if None, then shift = window_size)
	 stride : Optional[int] = 1, # if split_windows, distance between single window elements (not recommended to change).
	 	See [tensorflow documentation](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#window) for details
	 axis : Optional[str] = 'bcf', # Dataset dimensionality. Either 'bcf' (batch, channel, features) or 'bfc' (batch, features, channel) (for RNNs)
	 plots : Optional[int] =  20, # number of subplots in indices plot 
	 verbose : Optional[bool] = True # verbosity mode 
)
```

### For git-lfs (.csv files)
* Deleting file from git-lfs
```
git filter-branch --index-filter \
    'git rm -rf --cached --ignore-unmatch [path_to_file]' HEAD
```
* clone / pull repository without LFS files
```
GIT_LFS_SKIP_SMUDGE=1 git clone SERVER-REPOSITORY
git lfs pull --include "*.csv" # to pull LFS files when needed
```
