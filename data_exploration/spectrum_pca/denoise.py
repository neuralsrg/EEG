import os
import argparse
from glob import glob

import noisereduce as nr
from scipy.io import wavfile


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Denoises audio files.')
    parser.add_argument('-d', '--dir', type=str, help='Directory with audio files')
    parser.add_argument('-o', '--output_dir', type=str, help='Directory for denoised audios', default='denoised_audios')
    parser.add_argument('-s', '--sens', type=float, help='Denoising sensitivity', default=1.0)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    files = glob(os.path.join(args.dir, '*.wav'))

    for filename in files:
        rate, data = wavfile.read(filename)
        reduced_noise = nr.reduce_noise(y=data, sr=rate, stationary=False, prop_decrease=args.sens)
        _, output_file = os.path.split(filename)
        output_file = os.path.join(args.output_dir, output_file)
        wavfile.write(output_file, rate, reduced_noise)