import os
import pickle
import argparse
from glob import glob

import librosa
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis

def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    p, k = Phi.shape
    R = np.eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        u, s, vh = np.linalg.svd(np.dot(Phi.T, np.asarray(Lambda)**3 - (gamma/p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T, Lambda))))))
        R = np.dot(u, vh)
        d = np.sum(s)
        if i and d / d_old < tol:
            break
    return np.dot(Phi, R)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Computes principal components on audio files from a given directory.')
    parser.add_argument('-d', '--dir', type=str, help='Directory with audio files')
    parser.add_argument('-n', '--n_components', type=int, help='Number of principal components')
    parser.add_argument('-m', '--method', type=str, help='Factor analysis method. One of "pca"/"favarimax"/"faquartimax"/"pcavarimax".', default='pcavarimax')
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

    method = args.method
    if method == 'pca':
        transformer = PCA(n_components=args.n_components)
    elif method in ['favarimax', 'faquartimax']:
        transformer = FactorAnalysis(n_components=args.n_components, rotation=method[2:])
    elif method == 'none':
        transformer = FactorAnalysis(n_components=args.n_components)

    if method in ['pca', 'favarimax', 'faquartimax', 'none']:
        transformer.fit(spectrums.T)
        if method in ['varimax', 'quartimax']:
            with open('FactorAnalysis_model.pkl', 'wb') as handle:
                pickle.dump(transformer, handle)
        data = np.concatenate((transformer.mean_[None, ...], transformer.components_)).T
    elif method == 'pcavarimax':
        spectrums = spectrums.T
        spec_mean = spectrums.mean(axis=0)
        spectrums -= spec_mean

        U, Sigma, V_T = np.linalg.svd(spectrums, full_matrices=False)
        U, Sigma, V_T = U[:, :args.n_components], Sigma[:args.n_components], V_T[:args.n_components]

        C = np.sqrt(spectrums.shape[0] - 1)
        U_tilde = C * U
        L_T_tilde = 1/C * Sigma.reshape(-1, 1) * V_T
        L_T_rot = varimax(L_T_tilde.T).T
        data = np.concatenate((spec_mean[None, ...], L_T_rot)).T
    else:
        raise AttributeError('Unknown method!')

    columns = ['mean'] + [f'component_{i+1}' for i in range(args.n_components)]

    pd.DataFrame(data, columns=columns).to_csv(args.output_file, index=False)