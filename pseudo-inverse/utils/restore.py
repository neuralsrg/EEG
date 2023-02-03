import numpy as np
import librosa


def restore(D, frame_size, hop_length, epochs=100, window='hann'):
    
    D = np.concatenate((np.zeros((D.shape[0], 1)), D, np.zeros((D.shape[0], 1))), axis=1)
    mag, _ = librosa.magphase(D)
    #mag = np.abs(D)
    
    phase = np.exp(1.j * np.random.uniform(0., 2*np.pi, size=mag.shape))
    x_ = librosa.istft(mag * phase, hop_length=hop_length, center=False, window=window)
    
    for i in range(epochs):
        _, phase = librosa.magphase(librosa.stft(x_, n_fft=frame_size, hop_length=hop_length, center=False,
                                                 window=window))
        x_ = librosa.istft(mag * phase, hop_length=hop_length, center=False, window=window)
        
    return x_[hop_length:-hop_length]


def restore_matrix(M, frame_size, hop_length, epochs=100, window='hann'):
    freq_bins = frame_size // 2 + 1
    M = M.reshape(-1, freq_bins).T

    return restore(M, frame_size, hop_length, epochs=epochs, window=window)