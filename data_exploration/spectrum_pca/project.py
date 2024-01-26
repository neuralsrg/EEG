import os
import pickle
import argparse
from glob import glob

import librosa
import numpy as np
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Projects sound spectrums onto given basis vectors.')
    parser.add_argument('-d', '--dir', type=str, help='Directory with audio files')
    parser.add_argument('-c', '--components_path', type=str, help='Path to csv file with principal components')
    parser.add_argument('-m', '--method', type=str, help='Factor analysis method. One of "pca"/"favarimax"/"faquartimax"/"pcavarimax".', default='pcavarimax')
    parser.add_argument('-sr', '--sampling_rate', type=int, help='Audio sampling rate', default=44100)
    parser.add_argument('-fs', '--frame_size', type=int, help='Frame size (n_fft) for STFT', default=2048)
    parser.add_argument('-hop', '--hop_size', type=int, help='Hop size for STFT', default=512)
    parser.add_argument('-o', '--output_dir', type=str, help='Directory where projection coefficients will be saved', default='projections')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    df = pd.read_csv(args.components_path)
    mean = df['mean'].to_numpy()
    components = df.iloc[:, 1:].to_numpy()
    components = np.linalg.pinv(components).T if (args.method == 'pcavarimax') else components

    files = glob(os.path.join(args.dir, '*.wav'))
    for filename in files:

        sound = librosa.load(filename, sr=args.sampling_rate, mono=True)[0]
        spectrum = librosa.stft(sound, n_fft=args.frame_size, hop_length=args.hop_size, center=True)
        spectrum = np.abs(spectrum)

        if args.method in ['pca', 'pcavarimax']:
            try:
                coefs = (spectrum.T - mean[None, ...]) @ components  # (time, n_components)
            except:
                print('Shapes did not match. Use the same STFT parameters for computing components and projecting spectrums.')
                exit(0)
        elif args.method in ['favarimax', 'faquartimax', 'none']:
            with open('FactorAnalysis_model.pkl', 'rb') as handle:
                transformer = pickle.load(handle)
                coefs = transformer.transform(spectrum.T)
        else:
            raise AttributeError('Unknown method!')

        columns = [f'component_{i+1}' for i in range(coefs.shape[1])]

        _, output_file = os.path.split(filename)
        output_file = output_file.split('.')[0] + '.csv'
        output_file = os.path.join(args.output_dir, output_file)
        pd.DataFrame(coefs, columns=columns).to_csv(output_file, index=False)