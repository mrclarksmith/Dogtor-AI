# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 12:17:51 2022

@author: server
"""
import sounddevice as sd
from shutil import copy
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import pandas as pd
from MelGan_no_WN import max_min_f, save_signals  # generate_samples_audio


import matplotlib.pyplot as plt
import librosa
import librosa.display

import os
import tensorflow as tf
import numpy as np

FRAME_LENGTH = 32  # NUMBER OF TIME SAMPLES
FRAME_HEIGHT = 96  # Number of mel channels

DESIRED_SAMPLES = FRAME_LENGTH*256

LEARNING_RATE = 0.0005
BATCH_SIZE = 64
EPOCHS = 50
SAMPLES = 5120

FRAME_LENGTH = 32
FRAME_WIDTH = 96


def generate_stft(audios):
    generated_stfts = []
    for audio in audios:

        stft = librosa.stft(np.squeeze(audio), hop_length=256, win_length=512)
        generated_stfts.append(stft)

    return generated_stfts


def data_out(samples=SAMPLES, filename=False):
    dog_dir = "D:/python2/woof_friend/Dogtor-AI/ML/data/ML_SET/dog"
    files = _get_wav_files(dog_dir)
    x_train = process_data(files[:samples], return_list=filename)
    if filename:
        return x_train, files[:samples]
    else:
        return x_train


def _get_wav_files(directory):
    fu = [os.path.join(dp, f) for dp, dn, filenames in os.walk(directory) for f in filenames if
          os.path.splitext(f)[1].lower() == '.wav']
    return fu


def process_data(mylist, return_list=False):
    result1 = []  # Mel frequency
    num_items = len(mylist)
    print("generating mels", num_items)
    for i, item in enumerate(mylist):
        if i % 200 == 0:
            print(i)
        r1 = preprocess(item)
        result1.append(r1)

    if return_list:
        print("returninglist")
        return result1
    else:
        result1 = np.dstack(result1)  # convert list to np array
        result1 = np.rollaxis(result1, -1)  # bring last axis to front
        result1 = np.expand_dims(result1, -1)  # add channel of 1 for conv2d to work
        # result1 = result1[:,(100-FRAME_WIDTH):,::]
        return result1


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


def _normalize(x, a=0, b=1):
    x = ((b-a)*(x - x.min()) / (x.max() - x.min())) + (a)
    return x


def _power_to_db(S):
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB


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


def graph3(f1, f2, f3):
    fig, (ax1, ax2, ax3) = plt.subplots(3,)

    img1 = librosa.display.specshow(f1, ax=ax1)
    img2 = librosa.display.specshow(librosa.amplitude_to_db(np.abs(f2),
                                                            ref=np.max),
                                    sr=22050, ax=ax2)
    img3 = plt.plot(range(len(f3)), f3,  'ro')
    fig.tight_layout()
    ax1.set(title='mel_spec original')
    ax2.set(title='mel_spec reconstructed')
    ax3.set(title='10 Latent Sapce')
    fig.show()


def graph(f1, f2, f3, f4):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,)

    img1 = librosa.display.specshow(f1, ax=ax1)
    img2 = librosa.display.specshow(f2, ax=ax2)
    img3 = librosa.display.specshow(librosa.amplitude_to_db(np.abs(f3),
                                                            ref=np.max),
                                    sr=22050, ax=ax3)
    img4 = plt.plot(range(len(f4)), f4,  'ro')
    fig.tight_layout()
    ax1.set(title='mel_spec original')
    ax2.set(title='mel_spec reconstructed')
    ax3.set(title='mel_spec from audio generated')
    ax4.set(title='10 Latent Sapce')
    fig.show()


def run_lite_model(X, interpreter):
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], X)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


def run_lite_model_vae_output(X, interpreter):
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    layer_details = interpreter.get_tensor_details()

    interpreter.set_tensor(input_details[0]['index'], X)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    encoding = interpreter.get_tensor(layer_details[59]['index'])

    return output_data, encoding

# Loading Models, Has to be done inside the module, or otherwise the graph breaks


def load_lite(directory, model_type, suffix=""):
    # Load TFLite model and allocate tensors.
    if model_type == "gan":
        lite_dir = os.path.join(directory, "gan_model_tflite.tflite")
    if model_type == "vae":
        lite_dir = os.path.join(directory, f"vae_model{suffix}_tflite.tflite")

    interpreter = tf.lite.Interpreter(model_path=lite_dir, experimental_preserve_all_tensors=True)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data.shape)

    return interpreter


def generate_samples_audio_chain(lite_model_vae_encoder, lite_model_vae_decoder, lite_model_gan, x, save=True):
    encoding = run_lite_model(x[np.newaxis, ...], lite_model_vae_encoder)
    reconstructed = run_lite_model(encoding, lite_model_vae_decoder)
    # code to load model
    mel_set = reconstructed.T
    print(mel_set.shape)
    mel_set = mel_set[np.newaxis, ...]
    # max_min_f(mel_set)

    # Generate audio_predict
    audio_generated = run_lite_model(np.squeeze(mel_set)[np.newaxis, ...], lite_model_gan)

    # Generate stft of both pred audio and new to graph to see the differnce without listening
    generated_stfts = generate_stft(audio_generated)

    graph(np.squeeze(x),
          np.squeeze(reconstructed),
          generated_stfts[0],
          np.squeeze(encoding)
          )

    # save audio
    if save:
        save_signals(audio_generated, "./model/")


def generate_point_audio(lite_model_vae_decoder,  lite_model_gan, encoding, save=True, return_audio=False):
    reconstructed = run_lite_model(encoding, lite_model_vae_decoder)
    # code to load model
    mel_set = reconstructed.T
    print(mel_set.shape)
    mel_set = mel_set[np.newaxis, ...]
    # max_min_f(mel_set)

    # Generate audio_predict
    audio_generated = run_lite_model(np.squeeze(mel_set)[np.newaxis, ...], lite_model_gan)

    # Generate stft of both pred audio and new to graph to see the differnce without listening
    generated_stfts = generate_stft(audio_generated)

    graph3(np.squeeze(reconstructed),
           generated_stfts[0],
           np.squeeze(encoding)
           )

    # save audio
    if save:
        save_signals(audio_generated, "./model/")
    if return_audio:
        return audio_generated


def point_finder(lite_model_vae_encoder, lite_model_vae_decoder, lite_model_gan, df):
    encoding_array = []
    reconstructed_array = []
    for index, row in df.iterrows():
        img = row['mel'][np.newaxis, ..., np.newaxis]

        encoding = run_lite_model(img, lite_model_vae_encoder)
        reconstructed = run_lite_model(encoding, lite_model_vae_decoder)
        encoding_array.append(encoding)
        reconstructed_array.append(reconstructed)

    df['encoding'] = encoding_array
    df['reconstructed'] = reconstructed_array

    return df


def get_kmeans(features, n_clusters=5, n_init=10, max_iter=300):
    kmeans = KMeans(
        init="random",
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=42
    )
    kmeans.fit(features)
    return kmeans


def _create_folder_if_it_doesnt_exist(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def copy_into_clusters(df, folder, cluster, number_of_items):
    _create_folder_if_it_doesnt_exist(folder)

    for item, row in df.iterrows():
        if row['cluster'] == cluster:
            rootname = row['filename']
            filename = rootname.split("\\")[-1]
            copy(os.path.join(rootname), os.path.join(str(folder), filename))
            print(filename)
        if item == number_of_items:
            break


def num_of_clusters(features):
    kmeans_kwargs = {
        "init": "random",
        "n_init": 6,
        "max_iter": 300,
        "random_state": 42,
    }
    silhouette_coefficients = []
    r1 = 2
    r2 = 15
    for k in range(r1, r2):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(features)
        score = silhouette_score(features, kmeans.labels_)
        silhouette_coefficients.append(score)

    plt.style.use("fivethirtyeight")
    plt.plot(range(r1, r2), silhouette_coefficients)
    plt.xticks(range(r1, r2))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.show()


# Load Models into memeory
number_of_audio_samples = 1
lite_model_vae_encoder = load_lite("model", "vae", suffix="_encoder")
lite_model_vae_decoder = load_lite("model", "vae", suffix="_decoder")
lite_model_gan = load_lite("model", "gan")
# x_train = data_out(10)
# g = generate_samples_audio_chain(lite_model_vae_encoder, lite_model_vae_decoder, lite_model_gan, x_train[3], save=False)


x_train, filename = data_out(400, filename=True)

df = pd.DataFrame()
df['mel'] = x_train
df['filename'] = filename

df = point_finder(lite_model_vae_encoder, lite_model_vae_decoder, lite_model_gan, df)
# x = np.squeeze(np.array(df['encoding'].to_list()))

# # num_of_clusters(x)
# k = get_kmeans(x)
# answer = k.fit_predict(x)
# df['cluster'] = answer

# copy_into_clusters(df, "./4", 4, number_of_items=100)


array = [[-3.0, -1.6774293,  0.5526688,  7.012168, -2.2925243,
         5.7915726,  -1.7413237,  3.5634975, -3.6460133,  -3.57509345]]
ar = np.array(array, dtype="float")
arc = ar.astype(np.float32)
generate_point_audio(lite_model_vae_decoder, lite_model_gan, arc, save=True)

audio = generate_point_audio(lite_model_vae_decoder, lite_model_gan,
                             arc, save=True, return_audio=True)
sd.play(np.squeeze(audio), samplerate=22050)
