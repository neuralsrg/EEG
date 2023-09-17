import os
import argparse
from glob import glob

import librosa
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Computes principal components on audio files from a given directory.')
    parser.add_argument('-d', '--dir', type=str, help='Directory with audio files')
    parser.add_argument('-n', '--n_components', type=int, help='Number of principal components')
    parser.add_argument('-sr', '--sampling_rate', type=int, help='Audio sampling rate', default=44100)
    parser.add_argument('-fs', '--frame_size', type=int, help='Frame size (n_fft) for STFT', default=2048)
    parser.add_argument('-hop', '--hop_size', type=int, help='Hop size for STFT', default=512)
    parser.add_argument('-o', '--output_file', type=str, help='File to save principal components', default='components.csv')
    args = parser.parse_args()

    files = glob(os.path.join(args.dir, '*.wav'))
    sounds = [librosa.load(f, sr=args.sampling_rate, mono=True)[0] for f in files]
    sounds = np.concatenate(sounds)

    spectrums = librosa.stft(sounds, n_fft=args.frame_size, hop_length=args.hop_size, center=True)
    spectrums = np.abs(spectrums)

    transformer = PCA(n_components=args.n_components)
    transformer.fit(spectrums.T)

    data = np.concatenate((transformer.mean_[None, ...], transformer.components_)).T
    columns = ['mean'] + [f'component_{i+1}' for i in range(args.n_components)]

    pd.DataFrame(data, columns=columns).to_csv(args.output_file, index=False)