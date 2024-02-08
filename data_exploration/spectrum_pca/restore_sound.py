import os
import argparse
from glob import glob

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Restores projections back to sound files.')
    parser.add_argument('-d', '--dir', type=str, help='Directory with projections')
    parser.add_argument('-c', '--components_path', type=str, help='Path to csv file with principal components')
    parser.add_argument('-m', '--method', type=str, help='Factor analysis method. One of "pca"/"favarimax"/"faquartimax"/"pcavarimax".', default='pcavarimax')
    parser.add_argument('-sr', '--sampling_rate', type=int, help='Audio sampling rate', default=44100)
    parser.add_argument('-fs', '--frame_size', type=int, help='Frame size (n_fft) for STFT', default=2048)
    parser.add_argument('-hop', '--hop_size', type=int, help='Hop size for STFT', default=512)
    parser.add_argument('-o', '--output_dir', type=str, help='Directory where restored sounds will be saved', default='restored_sounds')
    parser.add_argument('-high', '--highpass', type=float, help='High pass frequency', default=None)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    df = pd.read_csv(args.components_path)
    mean = df['mean'].to_numpy()
    components = df.iloc[:, 1:].to_numpy()

    files = glob(os.path.join(args.dir, '*.csv'))
    for filename in files:

        projections = pd.read_csv(filename).to_numpy()  # (time, n_components)
        if args.method in ['pca', 'favarimax', 'faquartimax', 'none', 'pcavarimax']:
            try:
                spectrum = components @ projections.T  # (n_freq, time)
                spectrum = spectrum + mean[..., None]
            except:
                print('Shapes did not match. Use the same STFT parameters for computing components and projecting spectrums.')
                exit(0)
        else:
            raise AttributeError('Unknown method!')
        
        if args.highpass is not None:
            freqs = np.arange(0, 1 + args.frame_size / 2) * args.sampling_rate / args.frame_size
            spectrum = spectrum * (args.highpass < freqs).astype(int)[:, None]

        sound = librosa.griffinlim(spectrum, hop_length=args.hop_size, n_fft=args.frame_size)

        _, output_file = os.path.split(filename)
        output_file = output_file.split('.')[0] + '.wav'
        output_file = os.path.join(args.output_dir, output_file)
        sf.write(output_file, sound, args.sampling_rate)