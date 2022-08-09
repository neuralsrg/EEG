import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
import pandas as pd
import mne
from typing import Optional, List, Set, Tuple, Sequence

class WindowGenerator():

  def __init__(self,
               paths : Sequence[str]) -> None:
      assert type(paths) == list
      self._mne_data = [mne.io.read_raw_edf(path) for path in paths]
      self._electrodes = [mne_data.get_data() for mne_data in self._mne_data]
      self._electrodes = np.concatenate(self._electrodes, axis=-1)
      self._labeled_data = mne.io.RawArray(self._electrodes,
                                           mne.create_info(self._mne_data[0].ch_names,
                                                           1006.04, ch_types='eeg'))
      self._label_channel = self._electrodes[-1]
      self._electrodes = self._electrodes[:-1]

      self._normalize = np.max(np.abs(self._electrodes), axis=-1)[..., np.newaxis]
      self._electrodes = self._electrodes / self._normalize
      msg = '\n\nAll the data was normalized. Refer to normalization coefficient as WindowGenerator().normalize'
      print(f"\x1b[32m{msg}\x1b[0m")

      self._df = pd.DataFrame(self._electrodes.T, columns=self._mne_data[0].ch_names[:-1])


  @property
  def normalize(self):
    return self._normalize


  def eeg_info(self) -> pd.DataFrame:
    """ Prints eeg data info """
    for mne_data in self._mne_data:
      print(mne_data.info, '\n')
    return self._df.describe().transpose().head(68)


  def print_inds(self):
    """ Prints labels """
    print(self._label_channel[self._non_zero_label_inds])

  
  def _plot_indices(self, n_subplots : Optional[int] = 10) -> None:
    """ Plots listen / repeat / noise indices """
    all_ind = np.concatenate([self._non_zero_label_inds, self._indices_noise])
    min_ind = np.min(all_ind)
    max_ind = np.max(all_ind)
    splits = np.linspace(min_ind, max_ind, n_subplots + 1, dtype=int)

    non_zero_labels = np.arange(self._label_channel.shape[0])[self._label_channel > 0]

    fig = plt.figure(figsize=(20, 3 * n_subplots))

    for i in range(n_subplots):
      ax = fig.add_subplot(n_subplots, 1, i + 1);

      label_dots = non_zero_labels[(non_zero_labels >= splits[i]) &\
                                   (non_zero_labels < splits[i + 1])]
      noise_dots = self._indices_noise[(self._indices_noise >= splits[i]) &\
                                       (self._indices_noise < splits[i + 1])]

      ax.scatter(label_dots, np.zeros_like(label_dots),
                 edgecolors='k', label='Labels', c='#2ca02c', marker='X')

      ax.scatter(noise_dots, np.ones_like(noise_dots),
                 edgecolors='k', label='Noise labels', c='#2bc14d', marker='D')
      ax.axis('tight')

      plt.ylim([-0.1, 1.1])
      plt.xticks(label_dots)
      plt.yticks([0, 1])
      plt.grid()
      
      if i == 0:
        plt.legend()


  def _generate_indices(self) -> None:
    """ Creates indices for data windows """
    def select(values, indices):
      if values[0] == values[1] + 10:
        return set(indices)
      else:
        return {}
    
    ind = np.arange(self._label_channel.shape[0])
    ind = ind[self._label_channel > 0]

    indices = set()

    for i in range(ind.shape[0] - 1):
      indices = indices.union(select(self._label_channel[ind[i : i + 2]], ind[i : i + 2]))

    self._non_zero_label_inds = np.array(sorted(list(indices)))
    assert self._non_zero_label_inds.shape[0] % 2 == 0
    info = ["Total number of intervals (before windowing): ",
            f"{self._non_zero_label_inds.shape[0] // 2}\n",
            "If you want to generate noise indices, this is an exact length ",
            "of np.array you have to pass to create_dataset()"]
    print(f"\x1b[32m{''.join(info)}\x1b[0m")


  def _generate_sequences(self,
                          plots : int,
                          skip_in_repeat : Optional[int] = 100,
                          verbose : Optional[bool] = True,
                          noise : Optional[bool] = True) -> bool:
    """ Creates data sequences """
    indices_listen = self._non_zero_label_inds[::2]
    indices_repeat = self._non_zero_label_inds[1::2] + skip_in_repeat
    
    arrays = [np.arange(i, i + self._length, dtype=int) for i in indices_listen]
    indices_listen = np.sort(np.concatenate(arrays, axis=0))

    arrays = [np.arange(i, i + self._length, dtype=int) for i in indices_repeat]
    indices_repeat = np.sort(np.concatenate(arrays, axis=0))
    
    if self._indices_noise is None:
      indices_noise = np.concatenate((np.arange(1000, 1000 + self._length),
                                      indices_repeat[:-self._length] + 6000),
                                      dtype=int)
      
      # first points in each noise interval
      self._indices_noise = indices_noise[::self._length]
    else:

      arrays = [np.arange(i, i + self._length, dtype=int) for i in self._indices_noise]
      indices_noise = np.sort(np.concatenate(arrays, axis=0))
      
    # must be 0 otherwise noise intervals overlap with label intervals!
    self._noise_max_label = np.max(self._label_channel[indices_noise])

    self._labels = self._label_channel[self._non_zero_label_inds[1::2]]
    n_events = self._labels.shape[0]
    sig_listen = np.empty((n_events, 0, self._length))
    sig_repeat = np.empty((n_events, 0, self._length))
    sig_noise = np.empty((n_events, 0, self._length))

    # so straight because of insufficient RAM problem...
    num_electrodes = self._raw_data.shape[0]
    for i in range(num_electrodes):

      sig_listen = np.append(sig_listen, self._raw_data[i][indices_listen].
          reshape(n_events, self._length)[:, np.newaxis, :], axis=1)
      
      sig_repeat = np.append(sig_repeat, self._raw_data[i][indices_repeat].
          reshape(n_events, self._length)[:, np.newaxis, :], axis=1)
      
      sig_noise = np.append(sig_noise, self._raw_data[i][indices_noise].
          reshape(n_events, self._length)[:, np.newaxis, :], axis=1)

    self._listen = sig_listen
    self._repeat = sig_repeat
    self._noise = sig_noise

    return self._check_noise(plots) if noise else True


  def _check_noise(self, plots : int) -> bool:
    """ Checks whether noise points overlap with labels """
    
    if self._noise_max_label:
      error = "\nNoise intervals overlap with labels. Please check them and set manually!"
      print(f"\x1b[31m{error}\x1b[0m")
      self._plot_indices(plots)
      return False
    return True


  def _split_windows(self) -> None:
    """ Splits data sequences into smaller windows """
    def window_ds(array : np.array) -> np.array: # shape (batch, channels, features)
      array = array.T # shape (features, channels, batch)
      ds = tf.data.Dataset.from_tensor_slices(array)
      win = ds.window(self._window_size,  # (num_windows, window_size, channels, batch)
                      shift=self._shift,
                      stride=self._stride,
                      drop_remainder=True)
      
      flatten = lambda x: x.batch(self._window_size, drop_remainder=True)
      win = win.flat_map(flatten)

      array = np.array(list(win.as_numpy_iterator()))
      array = np.moveaxis(array, 0, -1) # (window_size, channels, batch, num_windows)
      shape = (array.shape[0], array.shape[1], array.shape[2] * array.shape[3])
      array = array.reshape(shape) # (window_size, channels, batch)

      return array.T # (batch, channels, window_size)

    self._listen = window_ds(self._listen)
    self._repeat = window_ds(self._repeat)
    self._noise = window_ds(self._noise)

  
  def _verb(self, listen_repeat_noise : Optional[Tuple[bool, bool, bool]]) -> None:
    """ Prints additional info """
    msg = np.array([
        f'listen.shape : {self._listen.shape}',
        f'repeat.shape : {self._repeat.shape}',
        f'noise.shape : {self._noise.shape}'
    ])
    print('\n'.join(msg[listen_repeat_noise]))


  def _move_axis(self) -> None:
    """ Changes dataset axis mode """
    self._listen = np.moveaxis(self._listen, -1, -2)
    self._repeat = np.moveaxis(self._repeat, -1, -2)
    self._noise = np.moveaxis(self._noise, -1, -2)

  
  def create_dataset(self,
                     event_length : Optional[int] = 300,
                     indices_noise : Optional[np.array] = None,
                     plot_indices : Optional[bool] = True,
                     skip_in_repeat : Optional[int] = 100,
                     listen_repeat_noise : Optional[Tuple[bool, bool, bool]] = [True, False, True],
                     train_val_test : Optional[Tuple[float, float, float]] = [.8, .2, 0],
                     batch_size : Optional[int] = 32,
                     split_windows : Optional[bool] = False,
                     channels : Optional[Sequence[int]] = [],
                     window_size : Optional[int] = 128,
                     shift : Optional[int] = None,
                     stride : Optional[int] = 1,
                     axis : Optional[str] = 'bcf',
                     plots : Optional[int] =  20,
                     verbose : Optional[bool] = True) -> None:
    """
    Creates datasets
    
    Args:
    event_length (optional) -- int. Length of the event. 300ms = 300 samples
    indices_noise (optional) -- np.array of first points in each noise interval of shape same as
      number of listen / repeat labels
    plot_indices (optional) -- bool. Whether to plot noise vs labeled indices plot
    skip_in_repeat (optional) -- samples to skip after repeat label. 100ms = 100 samples
    listen_repeat_noise (optional) -- List[bool, bool, bool]. Specifies whitch data to include in the dataset
      The first bool refers to listen, the second to repeat etc.
    train_val_test (optional) -- List[float, float, float]. Specifies train/val/test ratios respectively
    batch_size (optional) -- batch size
    split_windows (optional) -- bool. Whether to breake neural data sequences down into smaller ones
    channels (optional) -- array of electrode channel indices to use in dataset (e.g. np.arange(19, 68))
    window_size (optional) -- int. Used if split_windows == True. The length of the smaller windows
    shift (optional) -- int. Used if split_windows == True. Hop length used when creating smaller windows.
      If None, then smaller windows do not overlap.
    stride (optional) -- int. Used if split_windows == True. Determines the stride between input elements within a window.
      Not recommended to change!
    axis (optional) -- str. Either 'bcf' (batch, channels, features) or 'bfc' (batch, features, channels).
      'bfc' is usually used with RNNs. Use 'bcf' in most other cases.
    plots (optional) -- int. Number of subplots in label - noise plot
    verbose : Optional[bool] = True. Whether to print logging messages

    Returns:
    None.
    Generated datasets are stored in WindowGenerator.train, WindowGenerator.val and WindowGenerator.test
    """
    assert np.sum(train_val_test) == 1
    assert axis in ['bcf', 'bfc']

    self._length = event_length
    self._window_size = window_size
    self._shift = shift # The shift argument determines the number of input elements to shift between the start of each window
    self._stride = stride # The stride argument determines the stride between input elements within a window

    self._raw_data = self._electrodes if len(channels) == 0 else self._electrodes[channels]
    self._indices_noise = indices_noise

    self._generate_indices()
    if (not self._generate_sequences(plots, skip_in_repeat, verbose,
                                     noise=listen_repeat_noise[2])) and (listen_repeat_noise[2]):
      return

    if listen_repeat_noise[2]:
      assert self._non_zero_label_inds.shape[0] / 2 == self._indices_noise.shape[0]

    if plot_indices and listen_repeat_noise[2]:
      self._plot_indices(plots)

    if split_windows:
      self._split_windows()

    if axis == 'bfc':
      self._move_axis()

    if verbose:
      self._verb(listen_repeat_noise)

    self._listen_ds = tf.data.Dataset.from_tensor_slices(self._listen)
    self._repeat_ds = tf.data.Dataset.from_tensor_slices(self._repeat)
    self._noise_ds = tf.data.Dataset.from_tensor_slices(self._noise)

    num_examples = np.count_nonzero(listen_repeat_noise) * self._listen.shape[0]

    assert np.count_nonzero(listen_repeat_noise) >= 2
    noise_listen_repeat = np.array(listen_repeat_noise)[[2, 0, 1]]
    it = iter([0, 1, 2])
    labels = []
    for b in noise_listen_repeat:
      apnd = next(it) if b else 2
      labels.append(apnd)
    assert set(labels) == set([0, 1, 2])
    noise_lambda = lambda x: (x, labels[0])
    listen_lambda = lambda x: (x, labels[1])
    repeat_lambda = lambda x: (x, labels[2])

    self._listen_ds = self._listen_ds.map(listen_lambda, num_parallel_calls=tf.data.AUTOTUNE)
    self._repeat_ds = self._repeat_ds.map(repeat_lambda, num_parallel_calls=tf.data.AUTOTUNE)
    self._noise_ds = self._noise_ds.map(noise_lambda, num_parallel_calls=tf.data.AUTOTUNE)

    datasets = np.array([self._listen_ds, self._repeat_ds, self._noise_ds])
    datasets = datasets[listen_repeat_noise]

    dataset = datasets[0]
    if len(datasets) > 1:
      for ds in datasets[1:]:
        dataset = dataset.concatenate(ds)

    print('Total number of elements in the dataset:', num_examples)

    dataset = dataset.shuffle(num_examples)

    n_sets = np.count_nonzero(train_val_test)
    sizes = [int(num_examples * train_val_test[i]) for i in range(n_sets - 1)]
    sizes.append(num_examples - int(np.sum(sizes)))

    datasets = []
    for size in sizes:
      datasets.append(dataset.take(size))
      dataset = dataset.skip(size)

    sets = np.array(['train', 'val', 'test'])[np.arange(n_sets)]
    if verbose:
      msg = [f'{sets[i]} dataset contains {size} elements' for i, size in enumerate(sizes)]
      print('\n'.join(msg))

    print('\nRefer to datasets as:')
    msg = [f'\t WindowGenerator().{s}' for s in sets]
    print('\n'.join(msg))

    print('Note: If Noise == True then Noise is encoded with label 0')
    print('Second priority is always Listen and then Repeat')

    self._train = datasets[0].batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    if len(datasets) > 1:
      self._val = datasets[1].batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    if len(datasets) > 2:
      self._test = datasets[2].batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

    
  @property
  def train(self):
    return self._train


  @property
  def val(self) -> tf.data.Dataset:
    return self._val


  @property
  def test(self) -> tf.data.Dataset:
    return self._test
