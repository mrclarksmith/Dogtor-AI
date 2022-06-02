# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 23:46:56 2022
this is main
creates an audio out of generated mel spectrogram
@author: server
"""

"This is melgan from https://keras.io/examples/audio/melgan_spectrogram_inversion/"
'https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py'
"Alexsey Gromov"
import time

"define Custom Mel gan"
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_addons import layers as addon_layers
import random
import matplotlib.pyplot as plt
import librosa
import numpy as np
import os
import librosa.display
from timeit import default_timer as timer
from datetime import datetime
import soundfile as sf
import tensorboard




# Setting logger level to avoid input shape warnings
tf.get_logger().setLevel("ERROR")

FRAME_LENGTH = 32 #NUMBER OF TIME SAMPLES
FRAME_HEIGHT = 96 #Number of mel channels

# Defining hyperparameters
DESIRED_SAMPLES = FRAME_LENGTH*256

LEARNING_RATE_GEN = 1e-4
LEARNING_RATE_DISC = 1e-4
BATCH_SIZE = 32
DROP_OUT = 0.05 #dropout rate
mse = keras.losses.MeanSquaredError()
mae = keras.losses.MeanAbsoluteError()

# Splitting the dataset into training and testing splits
def _get_wav_files(directory):
    list_of_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(directory) for f in filenames if
                  os.path.splitext(f)[1].lower() == '.wav']
    return list_of_files
       



def _normalize(x, a=0, b= 1):
    x = ( (b-a)*( x- x.min()) / ( x.max() - x.min() ) ) +(a)
    return x

def _power_to_db(S):
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB

def _data_pad_random(item, frame_length):
    if item.shape[1] < frame_length:
        pad = int( (frame_length-item.shape[1])*random.random() )
        padded = np.pad(
                    item,(
                        (0,0),(pad,(frame_length-item.shape[1]-pad))
                        ),'constant',constant_values = (0)
                    )
        
    elif item.shape[1] > frame_length:
        before = int((item.shape[1]-frame_length)*random.random()*.3 )
        padded = item[:,before:frame_length+before]
            
    return padded    

def _data_pad(item, frame_length):
    if item.shape[1] > frame_length:
        item = item[:,:frame_length]
        return item
    elif item.shape[1] < frame_length:
        item = np.pad(
                    item,(
                        (0,0),(0,(frame_length-item.shape[1]))
                        ),'constant',constant_values = (0)
                    )
        return item
    return item

#https://github.com/tensorflow/tensorflow/issues/49238
def preprocess(filename, desired_samples = DESIRED_SAMPLES, load=True):
    sr = 22050
    n_fft = 1024
    hop_length = 256
    n_mfcc = FRAME_HEIGHT              
    window_type = 'hann' 
    feature = 'mel'     
    n_mels=FRAME_HEIGHT
    frame_lenght= FRAME_LENGTH #NUMBER OF TIME SAMPLES
    frame_height = FRAME_HEIGHT #NUMBER OF MELS
    freq_max = int(22050/2)
    freq_min = 125
    if load:        
        audio , _ = librosa.load(filename, res_type='kaiser_fast', sr=sr, mono=True)
        audio  = np.trim_zeros(audio)
        audio = audio[:desired_samples] 
        audio = np.pad(audio, (0, desired_samples-audio.size) , mode='constant')
    else:
        audio = filename
    # audio = tf.audio.decode_wav(tf.io.read_file(filename), 1, DESIRED_SAMPLES).audio
    audio = librosa.util.normalize(audio)
    mel = librosa.feature.melspectrogram(audio,
                                           sr=sr, 
                                           n_fft=n_fft,
                                           hop_length=hop_length, 
                                           n_mels=n_mels, 
                                           htk=False, 
                                           fmin=freq_min
                                   )

    # Taking the magnitude of the STFT output  
    mel = _power_to_db(mel)
    mel = _normalize(mel)
    mel = _data_pad(mel, frame_lenght)
    mel = mel.T
    return mel, audio

def get_data(mylist):
    result1 = []
    result2 = []
    
    for item in mylist:
        r1, r2 = preprocess(item)
        result1.append(r1)
        result2.append(r2)
    return result1, result2


class ReflectionPad(keras.layers.Layer):
    def __init__(self, padding):
        self.padding = padding
        super().__init__()
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "padding": self.padding,
        })
        return config    
    
        
    def call(self,inputs):
        return tf.pad(inputs, [[0,0],[self.padding,self.padding], [0,0]], "REFLECT")
    
# Creating the residual stack block
def residual_stack(input, filters, dilation):
    """Convolutional residual stack with weight normalization.

    Args:
        filter: int, determines filter size for the residual stack.

    Returns:
        Residual stack output.
    """
    x = layers.LeakyReLU(0.2)(input)
    x = addon_layers.WeightNormalization(
        layers.Conv1D(filters, kernel_size=3, dilation_rate=dilation, padding="same"),data_init=False)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = addon_layers.WeightNormalization(
        layers.Conv1D(filters, kernel_size=1, dilation_rate=1, padding="same"),data_init=False )(x)
    
    
    shotcut = addon_layers.WeightNormalization(
            layers.Conv1D(filters, kernel_size=1, dilation_rate=1),data_init=False)(input)
    add = layers.Add()([x, shotcut])    
    return add

def create_generator(input_shape, n_residual_layers=3):
    ratios = [8,8,2,2]
    conv_dim = 256
    
    inp = keras.Input(input_shape)
    # x = MelSpec()
    x = ReflectionPad(3)(inp)
    x = addon_layers.WeightNormalization(
        layers.Conv1D(512, 7), data_init=False)(x)
    
    for i, r in enumerate(ratios):
        x = layers.LeakyReLU(0.2)(x)
        x = addon_layers.WeightNormalization(
            layers.Conv1DTranspose(conv_dim, kernel_size=r*2, strides=r, padding="same"),
            data_init=False )(x)
        
        for j in range(n_residual_layers):
            x = residual_stack(x, conv_dim, dilation=3**j)   
            
        conv_dim //= 2

    x = layers.LeakyReLU(0.2)(x)
    # x = ReflectionPad(3)(x)
    x = addon_layers.WeightNormalization(
        layers.Conv1D(1, 7, activation="tanh", name='conv1D_generator', padding="same"),data_init=False 
    )(x)
    return keras.Model(inp, x)


def discriminator_block(input):
    conv1 = addon_layers.WeightNormalization(
        layers.Conv1D(16, 15, 1, "same"), data_init=False
    )(input)
    lrelu1 = layers.LeakyReLU(0.2)(conv1)
    drop1 = layers.Dropout(rate=DROP_OUT)(lrelu1)
    conv2 = addon_layers.WeightNormalization(
        layers.Conv1D(64, 41, 4, "same", groups=4), data_init=False
    )(drop1)
    lrelu2 = layers.LeakyReLU(0.2)(conv2)
    conv3 = addon_layers.WeightNormalization(
        layers.Conv1D(256, 41, 4, "same", groups=16), data_init=False
    )(lrelu2)
    lrelu3 = layers.LeakyReLU(0.2)(conv3)
    conv4 = addon_layers.WeightNormalization(
        layers.Conv1D(1024, 41, 4, "same", groups=64), data_init=False
    )(lrelu3)
    lrelu4 = layers.LeakyReLU(0.2)(conv4)
    # drop2 = layers.Dropout(rate=0.2)(lrelu4)
    conv5 = addon_layers.WeightNormalization(
        layers.Conv1D(1024, 41, 4, "same", groups=256), data_init=False
    )(lrelu4)
    
    lrelu5 = layers.LeakyReLU(0.2)(conv5)
    conv6 = addon_layers.WeightNormalization(
        layers.Conv1D(1024, 5, 1, "same"), data_init=False
    )(lrelu5)
    lrelu6 = layers.LeakyReLU(0.2)(conv6)
    drop3 =  layers.Dropout(DROP_OUT)(lrelu6)
    conv7 = addon_layers.WeightNormalization(
        layers.Conv1D(1, 3, 1, "same", name='conv1d_final_disc'), data_init=False
    )(drop3)
    return [lrelu1, lrelu2, lrelu3, lrelu4, lrelu5, lrelu6, conv7]



def create_discriminator(input_shape):
    inp = keras.Input(input_shape)
    out_map1 = discriminator_block(inp)
    pool1 = layers.AveragePooling1D(4, strides=2, padding="same")(inp)
    out_map2 = discriminator_block(pool1)
    pool2 = layers.AveragePooling1D(4, strides=2, padding="same")(pool1)
    #drop =  layers.Dropout(rate=DROP_OUT)(pool2)
    out_map3 = discriminator_block(pool2)
    return keras.Model(inp, [out_map1, out_map2, out_map3])


# We use a /dynamic/ input shape for the generator since the model is fully convolutional
generator = create_generator((FRAME_LENGTH,FRAME_HEIGHT))
generator.summary()


# We use a dynamic input shape for the discriminator
# This is done because the input shape for the generator is unknown
discriminator = create_discriminator((DESIRED_SAMPLES,1))
discriminator.summary()

@tf.function
def generator_loss(fake_pred):
    """Loss function for the generator.

    Args:
        fake_pred: Tensor, output of the generator prediction passed through the discriminator.

    Returns:
        Loss for the generator.
    """
    gen_loss = 0
    for i in range(len(fake_pred)):
        gen_loss += -1* tf.reduce_mean(fake_pred[i][-1])

    return gen_loss

@tf.function
def feature_matching_loss(real_pred, fake_pred):
    """Implements the feature matching loss.

    Args:
        real_pred: Tensor, output of the ground truth wave passed through the discriminator.
        fake_pred: Tensor, output of the generator prediction passed through the discriminator.

    Returns:
        Feature Matching Loss.
    """
    '''
    feat_weights = 4.0 / (args.n_layers_D + 1)
           D_weights = 1.0 / args.num_D'''
    feat_weight =  4.0 / (4+1)
    D_weights =  1 / 3
    wt = feat_weight * D_weights
    fm_loss = 0
    for i in range(len(fake_pred)):
        for j in range(len(fake_pred[i]) - 1):
            fm_loss += wt * tf.reduce_mean( mae(real_pred[i][j], fake_pred[i][j]))
            

    return fm_loss

@tf.function
def discriminator_loss(real_pred, fake_pred):
    """Implements the discriminator loss.

    Args:
        real_pred: Tensor, output of the ground truth wave passed through the discriminator.
        fake_pred: Tensor, output of the generator prediction passed through the discriminator.

    Returns:
        Discriminator Loss.
    """
    real_loss, fake_loss = 0, 0
    for i in range(len(real_pred)):
        real_loss += tf.reduce_mean(tf.nn.relu(tf.ones_like(real_pred[i][-1]) - real_pred[i][-1]))
        fake_loss += tf.reduce_mean(tf.nn.relu(tf.ones_like(fake_pred[i][-1]) + fake_pred[i][-1]))

    # Calculating the final discriminator loss after scaling
    disc_loss = real_loss + fake_loss
    return disc_loss


@tf.function
def mel_spectrogram_loss(fake_pred, real_predict):
    '''
    Return mel los of comparing short time L1 loss
    '''
    generated_stfts =  generate_stft(fake_pred)
    real_stfts = generated_stfts =  generate_stft(real_predict)
    mel_loss = 0
    for g_stft, r_stft in zip(generated_stfts, real_stfts):
        mel_loss += tf.reduce_mean(mae(g_stft, r_stft))
    return mel_loss 

class MelGAN(keras.Model):
    def __init__(self, generator, discriminator,  **kwargs):
        """MelGAN trainer class

        Args:
            generator: keras.Model, Generator model
            discriminator: keras.Model, Discriminator model
        """
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        # self.checkpoint = checkpoint

    def compile(
        self,
        gen_optimizer,
        disc_optimizer,
        generator_loss,
        feature_matching_loss,
        discriminator_loss,
        mel_spectrogram_loss,
    ):
        """MelGAN compile method.

        Args:
            gen_optimizer: keras.optimizer, optimizer to be used for training
            disc_optimizer: keras.optimizer, optimizer to be used for training
            generator_loss: callable, loss function for generator
            feature_matching_loss: callable, loss function for feature matching
            discriminator_loss: callable, loss function for discriminator
        """
        super().compile()

        # Optimizers
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer

        # Losses
        self.generator_loss = generator_loss
        self.feature_matching_loss = feature_matching_loss
        self.discriminator_loss = discriminator_loss

        # Trackers
        self.gen_loss_tracker = keras.metrics.Mean(name="gen_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="disc_loss")
        self.fm_loss_tracker = keras.metrics.Mean(name="fm_loss")
    
    @tf.function
    def train_step(self, batch):
        x_batch_train, y_batch_train = batch

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generating the audio wave
            gen_audio_wave = generator(x_batch_train, training=True)

            # Generating the features using the discriminator
            real_pred = discriminator(y_batch_train)
            fake_pred = discriminator(gen_audio_wave)

            # Calculat total loss
            # mel_loss =  mel_spectrogram_loss(fake_pred, real_pred)
            # Calculating the generator losses
            gen_loss = generator_loss(fake_pred)
            fm_loss = feature_matching_loss(real_pred, fake_pred)

            # Calculating final generator loss
            gen_fm_loss = gen_loss + 10. * fm_loss

            # Calculating the discriminator losses
            disc_loss = discriminator_loss(real_pred, fake_pred)

        # Calculating and applying the gradients for generator and discriminator
        grads_gen = gen_tape.gradient(gen_fm_loss, generator.trainable_weights)
        grads_disc = disc_tape.gradient(disc_loss, discriminator.trainable_weights)
        self.gen_optimizer.apply_gradients(zip(grads_gen, generator.trainable_weights))
        self.disc_optimizer.apply_gradients(zip(grads_disc, discriminator.trainable_weights))

        self.gen_loss_tracker.update_state(gen_fm_loss)
        self.fm_loss_tracker.update_state(fm_loss)
        self.disc_loss_tracker.update_state(disc_loss)
        # self.mel_loss_tracker.update_state(mel_loss)

        return {
            "gen_loss": self.gen_loss_tracker.result(),
            "fm_loss": self.fm_loss_tracker.result(),
            "disc_loss": self.disc_loss_tracker.result(),
            # "mel_loss": self.mel_loss_tracker.result(),
        }

    def save(self, checkpoint, checkpoint_prefix):
            checkpoint.save(file_prefix=checkpoint_prefix)
            print("saved")
    
    def restore(self, checkpoint):
            checkpoint.restore(tf.train.latest_checkpoint('./training_checkpoints'))
            print("restored")


def get_optimizer(opt_type, lr_g=LEARNING_RATE_GEN, lr_d=LEARNING_RATE_DISC):
    if opt_type == "adam":
        gen_optimizer = keras.optimizers.Adam(
            LEARNING_RATE_GEN, beta_1=0.5, beta_2=0.9, clipnorm=1
        )
        disc_optimizer = keras.optimizers.Adam(
            LEARNING_RATE_DISC, beta_1=0.5, beta_2=0.9, clipnorm=1
        )
    if opt_type == "rectified_adam":
        import tensorflow_addons as tfa
        gen_optimizer = tfa.optimizers.RectifiedAdam(
            LEARNING_RATE_GEN, beta_1=0.9, beta_2=0.99, clipnorm=1
        )
        disc_optimizer = tfa.optimizers.RectifiedAdam(
            LEARNING_RATE_DISC, beta_1=0.9, beta_2=0.99, clipnorm=.99
        )

    if opt_type == "SGD":
        gen_optimizer = keras.optimizers.SGD(
            LEARNING_RATE_GEN, 
        )
        disc_optimizer = keras.optimizers.SGD(
            LEARNING_RATE_DISC,
        )
    
    return gen_optimizer, disc_optimizer

def gen_data(wavs):
    print("generating mels", len(wavs))
    X_train, y_train = get_data(wavs,)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train, dtype=np.float32)
    y_train = y_train[...,  np.newaxis]
    
    return X_train, y_train

def init_checkpoint(gen_optimizer, disc_optimizer):

    
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
                                     discriminator_optimizer=disc_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    
    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #         filepath='./model/model_save{epoch:02d}-{disc_loss:.4f}{gen_loss:.4f}.h5',
    #         save_weights_only=True,
    #         monitor='disc_loss',
    #         mode='auto',
    #         save_freq=BATCH_SIZE,
    #         save_best_only=True,
    #         )

    return checkpoint, checkpoint_prefix,

def compile_model(mel_gan, gen_optimizer, disc_optimizer):
    mel_gan.compile(
        gen_optimizer,
        disc_optimizer,
        generator_loss,
        feature_matching_loss,
        discriminator_loss,
        mel_spectrogram_loss,
        )
    return mel_gan

def fit(mel_gan, X_train, y_train, epochs=10, batchsize=BATCH_SIZE):
    start = timer()
    logdir="./logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    mel_gan.fit(X_train,
                y= y_train,
                batch_size=BATCH_SIZE,
                epochs=epochs,
                verbose='auto',
                callbacks=[tensorboard_callback],
                )
    
    end = timer()
    print("Time to train:", end-start)
    return mel_gan

def audio_to_mel(wavs, number_of_audio_samples=1):
    mel_set = []
    audios = []
    for i in range(number_of_audio_samples):        
        wav = sys_random.choice(wavs)
        # print(wav)
        mel, audio = preprocess(wav, load=True)
        mel_set.append(mel)
        audios.append(audio)
        # print(mel.shape,"mel_shape")
        
    # fig, ax = plt.subplots(number_of_audio_samples,1)
    
    # if number_of_audio_samples == 1:
    #     librosa.display.specshow(mel_set[0].T, ax=ax)
    # else:    
    #     for i in range(number_of_audio_samples):
    #         librosa.display.specshow(mel_set[i].T, ax=ax[i])
    #         print("notch")
    print(mel_set[0].shape)
    return tf.convert_to_tensor(mel_set), audios


def generate_stft(audios):
    generated_stfts = []
    for audio in audios:
        
        stft = librosa.stft(np.squeeze(audio), hop_length=256, win_length=512 )
        generated_stfts.append(stft)
    
    return generated_stfts


def plot_real_vs_fake_stft(real, fake):
    number_of_rows =  len(real)
    fig2, ax = plt.subplots(number_of_rows,2)
    if number_of_rows == 1:
        ax[0].set(title = "real")
        ax[1].set(title = "fake")   
        
                             
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(real[0]),
                                                        ref=np.max),
                                sr=22050, x_axis='time', ax=ax[0] )
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(fake[0]),
                                                        ref=np.max),
                                sr=22050, x_axis='time', ax=ax[1] )
        
    
        
        
    else:
        ax[0,0].set(title = "real")
        ax[0,1].set(title = "fake")
        
    
        for i in range(number_of_rows):
            
            # librosa.display.specshow(real[i], y_axis='log', x_axis='time', ax=ax[i,0] )
            # librosa.display.specshow(fake[i], y_axis='log', x_axis='time', ax=ax[i,1] )
    
                         
            librosa.display.specshow(librosa.amplitude_to_db(np.abs(real[i]),
                                                            ref=np.max),
                                    sr=22050, x_axis='time', ax=ax[i,0] )
            librosa.display.specshow(librosa.amplitude_to_db(np.abs(fake[i]),
                                                            ref=np.max),
                                    sr=22050, x_axis='time', ax=ax[i,1] )
            
    fig2.suptitle('real vs fake stft of audio')
    cfm = plt.get_current_fig_manager()
    cfm.window.activateWindow()
    cfm.window.raise_()
    
def max_min_f(f):
    for i in range(f.shape[0]):
        u = f[i].max().round(3)
        l = f[i].min().round(3)
        print(f"f:{i}, min:{l}, max:{u}")
        
        
def save_signals(signals, save_dir, sample_rate=22050):
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, str(i) + ".wav")
        sf.write(save_path, signal, sample_rate)

sys_random = random.SystemRandom()
def generate_samples_audio(generator, number_of_audio_samples, save=True):
    #code to load model 
       
    wavs = _get_wav_files(r"D:/python2/woof_friend/Dogtor-AI/ML/data/ML_SET/dog")

    #trim aduio & pad at the end
    #Generate Mel 
    mel_set, real_audios = audio_to_mel(wavs, number_of_audio_samples)
    print(mel_set.shape)
    print(mel_set.numpy().shape)
    
    # mel_set = mel_set *0.8
    max_min_f(mel_set.numpy())
    # for x in range(mel_set.numpy().shape[0]):
    
    #Generate audio_predict
    audio_generated = generator.predict(mel_set, steps=2, batch_size=32, verbose=1)
      
    #Generate stft of both pred audio and new to graph to see the differnce without listening
    generated_stfts =  generate_stft(audio_generated)
    real_stfts = generate_stft(real_audios)
    
    plot_real_vs_fake_stft(real_stfts, generated_stfts, )
    
    #save audio    
    if save:
        save_signals(audio_generated,"./model/")
    
    # fake_res = discriminator.predict(audio_generated)
    


def Mel_gan_load_model_from_checkpoint():
    mel_gan = MelGAN(generator, discriminator)
    gen_optimizer, disc_optimizer = get_optimizer(opt_type="adam", lr_g=LEARNING_RATE_GEN, lr_d=LEARNING_RATE_DISC)
    checkpoint, checkpoint_prefix = init_checkpoint(gen_optimizer, disc_optimizer)
    mel_gan.restore(checkpoint) # msust do compile_model after
    mel_gan = compile_model(mel_gan, gen_optimizer, disc_optimizer)
    return mel_gan, generator


from autoencoder import VAE
from generate import get_all_points, get_mels, graph


def _test_mel(generator, vae, mels, audio_num=9):
    features  = get_all_points(mels, vae)
    f = features[1]
    mel_spectrogram = vae.decoder.predict(f[audio_num][np.newaxis,...])
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
    audio = generator.predict(m_s)
    
    mel_generated, _ = preprocess(np.squeeze(audio),load=False)
    print(np.array(mel_generated).shape)
    graph(np.squeeze(mels[audio_num]), mel_spectrogram, np.squeeze(mel_generated).T, f[audio_num])
    
    # save_signals([audio], "./mel_audio/")
    
def main():
    wavs = _get_wav_files(r"D:/python2/woof_friend/Dogtor-AI/ML/data/ML_SET/dog")
    print(f"Number of audio files: {len(wavs)}")
    
    # X_train, y_train = gen_data(wavs[:50])
    mel_gan, generator = Mel_gan_load_model_from_checkpoint()
    vae = VAE.load("model_no_eager_1")
    mels = get_mels(50)
    _test_mel(generator, vae, mels, audio_num=10)

    # mel_gan = MelGAN(generator, discriminator)
    
    # gen_optimizer, disc_optimizer = get_optimizer(opt_type="adam", lr_g=LEARNING_RATE_GEN, lr_d=LEARNING_RATE_DISC)
    # checkpoint, checkpoint_prefix = init_checkpoint(gen_optimizer, disc_optimizer)
    # mel_gan.restore(checkpoint) # msust do compile_model after
    # mel_gan = compile_model(mel_gan, gen_optimizer, disc_optimizer)
    # # mel_gan = fit(mel_gan, X_train, y_train, epochs=550, batchsize=64)
    # # mel_gan.save(checkpoint, checkpoint_prefix)
    
    
    number_of_audio_samples=1
    generate_samples_audio(generator, number_of_audio_samples, save=True)
    #'https://paperswithcode.com/method/wgan-gp-loss' 
    
# def restore_sessions():
#     seg_graph = tf.Graph()
#     sess = tf.Session(graph=seg_graph)
#     K.set_session(sess)

if __name__ == "__main__":
    main()
    pass
    
    
    