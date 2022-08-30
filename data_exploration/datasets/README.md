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

3. Now you can create dataset with `generator.create_dataset()`
```
Sequence generating parameters:

event_length (optional) -- int. Length of the event. 300ms = 300 samples
indices_noise (optional) -- np.array of first points in each noise interval of shape same as
  number of listen / repeat labels
plot_indices (optional) -- bool. Whether to plot noise vs labeled indices plot
skip_in_repeat (optional) -- samples to skip after repeat label. 100ms = 100 samples
listen_repeat_noise (optional) -- List[bool, bool, bool]. Specifies whitch data to include in the dataset
  The first bool refers to listen, the second to repeat etc.
channels (optional) -- array of electrode channel indices to use in dataset (e.g. np.arange(19, 68))


Wavelet transform parameters:

apply_cwt (optional) -- bool, whether to apply continious wavelet transform
wavelet_name (optional) -- str, wavelet name (see: https://pywavelets.readthedocs.io/en/latest/ref/cwt.html#continuous-wavelet-families)
cwt_channel (optional) -- int, which channel to use to perform wavelet transform
normalize_cwt (optional) -- bool, whether to normalize cwt matrices
plot_cwt (optional) -- bool, whether to plot noise / listen / repeat cwt graphs
cwt_inds_plot (optional) -- sequence of int, if plot_cwt == True, which samples from dataset to use for plotting 


Dataset generating parameters:

train_val_test (optional) -- List[float, float, float]. Specifies train/val/test ratios respectively
shuffle_before_splitting (optional) -- bool. Whether to shuffler data before splitting into
  train / val / test or after
batch_size (optional) -- batch size
axis (optional) -- str. Either 'bcf' (batch, channels, features) or 'bfc' (batch, features, channels).
  'bfc' is usually used with RNNs. Use 'bcf' in most other cases.


Windowing parameters: (not recommended when using wavelet transform)

split_windows (optional) -- bool. Whether to breake neural data sequences down into smaller ones
window_size (optional) -- int. Used if split_windows == True. The length of the smaller windows
shift (optional) -- int. Used if split_windows == True. Hop length used when creating smaller windows.
  If None, then smaller windows do not overlap.
stride (optional) -- int. Used if split_windows == True. Determines the stride between input elements within a window.
  Not recommended to change!


Verbosity parameters:

plots (optional) -- int. Number of subplots in label - noise plot
verbose : Optional[bool] = True. Whether to print logging messages
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
