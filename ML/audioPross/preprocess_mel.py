# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 21:38:12 2022

@author: server
"""
# Main preprocess operation for testing mells
import os
import librosa
import numpy as np
import random
import librosa.display
import matplotlib.pyplot as plt


FRAME_LENGTH = 32  # NUMBER OF TIME SAMPLES
FRAME_HEIGHT = 96  # Number of mel channels

DESIRED_SAMPLES = FRAME_LENGTH*256

LEARNING_RATE = 0.0005
BATCH_SIZE = 64
EPOCHS = 50
SAMPLES = 5120

FRAME_LENGTH = 32
FRAME_WIDTH = 96


def _get_wav_files(directory):
    list_of_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(directory) for f in filenames if
                     os.path.splitext(f)[1].lower() == '.wav']
    return list_of_files


def preprocess(filename, desired_samples=DESIRED_SAMPLES, load=True):
    sr = 22050
    N_FFT = 1024
    HOP_LENGTH = N_FFT//4
    # n_mfcc = FRAME_HEIGHT
    # window_type = 'hann'
    # feature = 'mel'
    frame_lenght = FRAME_LENGTH  # NUMBER OF TIME SAMPLES
    frame_height = FRAME_WIDTH  # NUMBER OF MELS
    FMIN = 125
    if load:
        audio, _ = librosa.load(filename, res_type='kaiser_fast', sr=sr, mono=True)
        audio = np.trim_zeros(audio)
        audio = audio[:desired_samples]
        # audio = np.pad(audio, (0, desired_samples-audio.size) , mode='constant')
    else:
        audio = filename
    mel = librosa.feature.melspectrogram(audio, sr,
                                         n_fft=N_FFT,
                                         hop_length=HOP_LENGTH,
                                         n_mels=frame_height,
                                         htk=False,
                                         fmin=FMIN
                                         )

    # Taking the magnitude of the STFT output
    mel = _power_to_db(mel)
    mel = _normalize(mel)
    mel = _data_pad(mel, frame_lenght)

    return mel


def process_data(mylist):
    result1 = []  # Mel frequency
    num_items = len(mylist)
    print("generating mels", num_items)
    for i, item in enumerate(mylist):
        if i % 200 == 0:
            print(i)
        r1 = preprocess(item)
        result1.append(r1)

    result1 = np.dstack(result1)  # convert list to np array
    result1 = np.rollaxis(result1, -1)  # bring last axis to front
    result1 = np.expand_dims(result1, -1)  # add channel of 1 for conv2d to work
    # result1 = result1[:,(100-FRAME_WIDTH):,::]
    return result1


def _get_wav_files(directory):
    fu = [os.path.join(dp, f) for dp, dn, filenames in os.walk(directory) for f in filenames if
          os.path.splitext(f)[1].lower() == '.wav']
    return fu


def _normalize(x, a=0, b=1):
    x = ((b-a)*(x - x.min()) / (x.max() - x.min())) + (a)
    return x


def _power_to_db(S):
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB


def _data_pad_random(item, frame_length):
    if item.shape[1] < frame_length:
        pad = int((frame_length-item.shape[1])*random.random())
        padded = np.pad(
            item, (
                (0, 0), (pad, (frame_length-item.shape[1]-pad))
            ), 'constant', constant_values=(0)
        )

    elif item.shape[1] > frame_length:
        before = int((item.shape[1]-frame_length)*random.random()*.3)
        padded = item[:, before:frame_length+before]

    return padded


def _data_pad(item, frame_length):
    if item.shape[1] > frame_length:
        item = item[:, :frame_length]
        return item
    elif item.shape[1] < frame_length:
        item = np.pad(
            item, (
                (0, 0), (0, (frame_length-item.shape[1]))
            ), 'constant', constant_values=(0)
        )
        return item
    return item


def get_mels(num_files=None):
    dog_dir = "D:/python2/woof_friend/Dogtor-AI/ML/data/ML_SET/dog"
    files = _get_wav_files(dog_dir)
    mel_spectrograms = process_data(files[:num_files])
    return mel_spectrograms


def graph(f1, f2, f3):
    fig, (ax1, ax2, ax3) = plt.subplots(3,)

    img1 = librosa.display.specshow(f1, ax=ax1)
    img2 = librosa.display.specshow(f2, ax=ax2)
    img3 = plt.plot(range(len(f3)), f3,  'ro')
    fig.tight_layout()
    ax1.set(title='mel_spec original')
    ax2.set(title='mel_spec reconstructed')
    ax3.set(title='10 Latent Sapce')
    fig.show()
