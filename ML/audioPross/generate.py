import os

import numpy as np
import soundfile as sf
import random

import matplotlib.pyplot as plt
import librosa
import librosa.display
from autoencoder import VAE
from MelGan import Mel_gan_load_model_from_checkpoint, _normalize, _data_pad

# from preprocess_tf import PreprocessData
from train import process_data, _get_wav_files, preprocess

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN

HOP_LENGTH = 256
SAVE_DIR_ORIGINAL = "samples/original/"
SAVE_DIR_GENERATED = "samples/generated/"
# MIN_MAX_VALUES_PATH = "/home/valerio/datasets/fsdd/min_max_values.pkl"



def graph(f1,f2,f3,f4):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,)
    
    img1 = librosa.display.specshow(f1, ax=ax1)
    img2 = librosa.display.specshow(f2, ax=ax2)
    img2 = librosa.display.specshow(f3, ax=ax3)
    img3 = plt.plot(range(len(f4)), f4,  'ro')
    fig.tight_layout()
    ax1.set(title='mel_spec original')
    ax2.set(title='mel_spec reconstructed')
    ax3.set(title='mel_spec from audio generated')
    ax4.set(title='10 Latent Sapce')
    fig.show()



def mel_s(spectrograms,
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


    
def get_mels(num_files=None):
    dog_dir = "D:/python2/woof_friend/Dogtor-AI/ML/data/ML_SET/dog"
    files = _get_wav_files(dog_dir)  
    mel_spectrograms =  process_data(files[:num_files])
    return mel_spectrograms

def get_all_points(mel_spectrograms, autoencoder):
    mels_predicted =autoencoder.reconstruct(mel_spectrograms)
    return mels_predicted

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

def num_of_clusters(features):
    kmeans_kwargs = {
    "init": "random",
    "n_init": 6,
    "max_iter": 300,
    "random_state": 42,
    }
    silhouette_coefficients = []
    for k in range (2,11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(features)
        score =  silhouette_score(features, kmeans.labels_)
        silhouette_coefficients.append(score)
    
    plt.style.use("fivethirtyeight")
    plt.plot(range(2, 11), silhouette_coefficients)
    plt.xticks(range(2, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.show()


def plot_clusters(scaled_features):
    # Instantiate k-means and dbscan algorithms
    kmeans = KMeans(n_clusters=5)
    dbscan = DBSCAN(eps=0.1)
    
    # Fit the algorithms to the features
    kmeans.fit(scaled_features)
    dbscan.fit(scaled_features)
    
    # Compute the silhouette scores for each algorithm
    kmeans_silhouette = silhouette_score(scaled_features, kmeans.labels_).round(3)
    # dbscan_silhouette = silhouette_score(scaled_features, dbscan.labels_).round(3)
    # Plot the data and cluster silhouette comparison
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(8, 6), sharex=True, sharey=True
    )
    fig.suptitle("Clustering Algorithm Comparison: Crescents", fontsize=16)
    fte_colors = {
        0: "#008fd5",
        1: "#fc4f30",
    }
    # The k-means plot
    km_colors = [fte_colors[label] for label in kmeans.labels_]
    ax1.scatter(scaled_features[:, 0], scaled_features[:, 1], c=km_colors)
    ax1.set_title(
        f"k-means\nSilhouette: {kmeans_silhouette}", fontdict={"fontsize": 12}
    )
    
    # The dbscan plot
    # db_colors = [fte_colors[label] for label in dbscan.labels_]
    # ax2.scatter(scaled_features[:, 0], scaled_features[:, 1], c=db_colors)
    # ax2.set_title(
    #     f"DBSCAN\nSilhouette: {dbscan_silhouette}", fontdict={"fontsize": 12}
    # )
    plt.show()


def generate_random_mel_features(n=10, mu=0, sigma=1.5):
    nums = [] 
    for i in range(n): 
        temp = random.gauss(mu, sigma) 
        nums.append(temp) 
    return np.array(nums)



def _test():
    x = generate_random_mel_features()
    print(x.max())
    print(x.min()) 
    
def _create_folder_if_it_doesnt_exist(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
def save_signals(signals, save_dir, sample_rate=22050):
    # _create_folder_if_it_doesnt_exist(save_dir)
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, "saved_audio"+str(i) + ".wav")
        sf.write(save_path, signal, sample_rate)

def bark_generator(vae, generator, param=None):

    # Generate Mel Spectrogram from Random Parameters 
    # Random Parameters
    bark_mel_parameters = generate_random_mel_features(n=10) #pretrained model parameters
    mel_spectrogram = vae.decoder.predict(bark_mel_parameters[np.newaxis,...])
    mel_spectrogram = np.squeeze(mel_spectrogram)

    
    librosa.display.specshow(mel_spectrogram)  
    
    #Load Audio generator from Mel Spectorgram

    audio = generator.predict(mel_spectrogram.T[np.newaxis,...])
    audio = np.squeeze(audio)
    save_signals([audio], "./random_mel_audio/")
    return audio



def _test_mel(gen_mel, autoencoder, mels, audio_num=9):
    features  = get_all_points(mels, autoencoder)
    f = features[1]
    mel_spectrogram = autoencoder.decoder.predict(f[audio_num][np.newaxis,...])
    FRAME_LENGTH = 32 #NUMBER OF TIME SAMPLES
    FRAME_HEIGHT = 96 #Number of mel channels
    mel_spectrogram = np.squeeze(mel_spectrogram)
    mel_spectrogram_n = _normalize(mel_spectrogram)
    mel_spectrogram_n = _data_pad(mel_spectrogram_n, FRAME_LENGTH)
    max_min_f(mel_spectrogram_n[np.newaxis,...])
    print(mel_spectrogram_n.shape, "mel1")
    
    m_s = mel_spectrogram_n.T
    # m_s = np.squeeze(mels[1].T)
    m_s = m_s[np.newaxis,...]
    # m_s = m_s[...,np.newaxis]
    print(m_s.shape)

    # m_s = tf.convert_to_tensor(m_s)
    audio = gen_mel.predict(m_s)
    
    mel_generated = preprocess(np.squeeze(audio), load=False)
    graph(np.squeeze(mels[audio_num]), mel_spectrogram, np.squeeze(mel_generated), f[audio_num])
    
    # save_signals([audio], "./mel_audio/")
    
    
def train_model():
    # initialise sound generator
    vae = VAE.load("model")
    # num_of_clusters(mel_features[1])
    # km = get_kmeans(mel_features[1])
    
def max_min_f(f):
    print(f.shape)
    for i in range(f.shape[0]):
        u = f[i].max().round(3)
        l = f[i].min().round(3)
        print(f"f:{i}, min:{l}, max:{u}")
            

def VAE_load(latent_space=10):
    autoencoder = VAE(
        input_shape=(FRAME_WIDTH, FRAME_LENGTH, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(1, 2, 2, 2, (2, 1)),
        latent_space_dim=latent_space
    )
    return autoencoder

    


if __name__ == "__main__":
    FRAME_LENGTH = 32
    FRAME_WIDTH= 96
    autoencoder = VAE_load(10)
    autoencoder.load_model_tf(save_folder="model_no_eager_test_full1") 
    mel_gan, gen_mel = Mel_gan_load_model_from_checkpoint()
    mels = get_mels(50)
    audio_num = 10
    _test_mel(gen_mel, autoencoder, mels, audio_num)


    pass

    # plot_clusters(mel_features[1])
    
    
    
    
    # mel = random.choice(mel_spectrograms) 
    # mels_predicted =vae.reconstruct(mel[np.newaxis, ...])
    
    # graph(np.squeeze(mel), np.squeeze(mel_predicted[0]), np.squeeze(mel_predicted[1]))
    
    


