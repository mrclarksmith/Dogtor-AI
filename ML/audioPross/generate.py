import os

import numpy as np
import soundfile as sf

import matplotlib.pyplot as plt
import librosa
import librosa.display
from autoencoder import VAE
from preprocess_tf import PreprocessData

import random


HOP_LENGTH = 256
SAVE_DIR_ORIGINAL = "samples/original/"
SAVE_DIR_GENERATED = "samples/generated/"
MIN_MAX_VALUES_PATH = "/home/valerio/datasets/fsdd/min_max_values.pkl"



def graph(f1,f2,f3):
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    
    img1 = librosa.display.specshow(f1, ax=ax1)
    img2 = librosa.display.specshow(f2, ax=ax2)
    img3 = plt.plot(range(len(f3)), f3,  'ro')
    fig.tight_layout()
    ax1.set(title='mel_spec original')
    ax2.set(title='mel_spec reconstructed')
    ax3.set(title='128 Latent Space')
    fig.show()







    x_train = []
    file_paths = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path) # (n_bins, n_frames, 1)
            x_train.append(spectrogram)
            file_paths.append(file_path)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # -> (3000, 256, 64, 1)
    return x_train, file_paths



def select_spectrograms(spectrograms,
                        file_paths,
                        min_max_values,
                        num_spectrograms=2):
    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms)
    sampled_spectrogrmas = spectrograms[sampled_indexes]
    file_paths = [file_paths[index] for index in sampled_indexes]
    sampled_min_max_values = [min_max_values[file_path] for file_path in
                           file_paths]
    print(file_paths)
    print(sampled_min_max_values)
    return sampled_spectrogrmas, sampled_min_max_values


def save_signals(signals, save_dir, sample_rate=22050):
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, str(i) + ".wav")
        sf.write(save_path, signal, sample_rate)


if __name__ == "__main__":
    # initialise sound generator
    vae = VAE.load("model")
    
    mel_spectrograms =  PreprocessData().process()
    
    mel = random.choice(mel_spectrograms)
    
    mel_predicted =vae.reconstruct(mel[np.newaxis, ...])
    graph(np.squeeze(mel), np.squeeze(mel_predicted[0]), np.squeeze(mel_predicted[1]))
    



