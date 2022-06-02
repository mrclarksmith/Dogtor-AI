# -*- coding: utf-8 -*-
"""
Created on Wed May 25 20:30:50 2022

@author: server
"""
"""
https://github.com/keras-team/keras-io/blob/master/examples/generative/vae.py
Title: Variational AutoEncoder
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/05/03
Last modified: 2020/05/03
Description: Convolutional Variational AutoEncoder (VAE) trained on MNIST digits.
"""

"""f
## Setup
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda 

from train import data_out
import datetime


"""
## Create a sampling layer
"""


"""
## Build the encoder
"""
EPOCHS = 500
FRAME_LENGTH = 32
FRAME_WIDTH= 96
rec_loss_multiplier = FRAME_LENGTH
latent_dim = 10
input_shape =(96, 32, 1)
#input_shape =(28, 28, 1)
SAMPLES = 3200



import matplotlib.pyplot as plt


import librosa
import librosa.display
def graph(f1,f2,f3):
    fig, (ax1, ax2, ax3) = plt.subplots(3,)
    
    img1 = librosa.display.specshow(f1, ax=ax1)
    img2 = librosa.display.specshow(f2, ax=ax2)
    img3 = plt.plot(range(len(f3)), f3,  'ro')
    fig.tight_layout()
    ax1.set(title='mel_spec original')
    ax2.set(title='mel_spec reconstructed')
    ax3.set(title='10 Latent Sapce')
    fig.show()


def reconstruct(vae, images):
    latent_representations = vae.encoder.predict(images)[1]
    reconstructed_images = vae.decoder.predict(latent_representations)
    return reconstructed_images, latent_representations


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, args):
        mu, log_variance = args
        # batch = K.shape(mu)[0]
        # dim = K.int_shape(mu)[1]
        epsilon = K.random_normal(shape=K.shape(mu), mean=0.,
                                  stddev=1.)
        sampled_point = mu + K.exp(log_variance / 2) * epsilon
        
        
        # sampled_point = tf.math.add(mu , tf.math.multiply( 
        #     tf.math.exp( tf.math.divide(log_variance, 2.)) , epsilon))
        return sampled_point

    def get_config(self):
        config = super().get_config()
        # config.update({"units": self.units})
        return config
    # def call(self, inputs):
    #     z_mean, z_log_var = inputs
    #     batch = tf.shape(z_mean)[0]
    #     dim = tf.shape(z_mean)[1]
    #     epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    #     return z_mean + tf.exp(0.5 * z_log_var) * epsilon





# encoder_inputs = keras.Input(shape=input_shape)
# x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
# x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.Flatten()(x)
# x = layers.Dense(16, activation="relu")(x)
# # z_mean = layers.Dense(latent_dim, name="z_mean")(x)
# # z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
# # z = Sampling()([z_mean, z_log_var])
# # encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
# # encoder.summary()

# """
# ## Build the decoder
# """

# latent_inputs = keras.Input(shape=(latent_dim,))
# x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
# x = layers.Reshape((7, 7, 64))(x)
# x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
# decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
# decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
# decoder.summary()
import os
def load_model(save_folder, f="tf"):
    print(save_folder)
    save_path = os.path.join(save_folder, f"full_model.{f}")
    print(save_path)
    model = tf.keras.models.load_model(save_path)
                               #custom_objects={'L__sample_point_from_normal_distribution': self.L__sample_point_from_normal_distribution})
    return model

def save_model(model, save_folder=".", f="tf"):
    _create_folder_if_it_doesnt_exist(save_folder)
    _save_model(model, save_folder, f=f)
    
    
def _save_model(model, save_folder=".",f="tf"):
    save_path = os.path.join(save_folder, f"full_model.{f}")
    model.save(save_path,save_format=f)


def _create_folder_if_it_doesnt_exist(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


class VAE_layers:
    """
    VAE represents a Deep Convolutional variational autoencoder architecture
    with mirrored encoder and decoder components.
    """

    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):
        self.input_shape = input_shape # [28, 28, 1]
        self.conv_filters = conv_filters # [2, 4, 8]
        self.conv_kernels = conv_kernels # [3, 5, 3]
        self.conv_strides = conv_strides # [1, 2, 2]
        self.latent_space_dim = latent_space_dim # 2
        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None
        self.model_output = None
        self.bottleneck = None

        self._build()

    


    def _build(self):
        self._build_encoder()
        self._build_decoder()
        # self._build_autoencoder()
     
    # def _build_autoencoder(self):
    #     model_input = self._model_input
    #     self._encoder_output = self.encoder(model_input)[0]
    #     self.model_output = self.decoder(self._encoder_output)
    #     self.model = Model(self._model_input, self.model_output, name="autoencoder")

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")        

    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck) # [1, 2, 4] -> 8
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        return Reshape(self._shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, x):
        """Add conv transpose blocks."""
        # loop through all the conv layers in reverse order and stop at the
        # first layer
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=1,
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer
    
## build Encoder ##
    def _build_encoder(self):
        self._model_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(self._model_input)
        z_mean, z_log_var, z = self._add_bottleneck(conv_layers)

        self.encoder = Model(self._model_input, [z_mean, z_log_var, z], name="encoder")

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")

    def _add_conv_layers(self, encoder_input):
        """Create all convolutional blocks in encoder."""
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        """Add a convolutional block to a graph of layers, consisting of
        conv 2d + ReLU + batch normalization.
        """
        layer_number = layer_index #+ 1
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_number}"
        )
        x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        return x    
    
    def _add_bottleneck(self, x):    
        """Flatten data and add bottleneck with Guassian sampling (Dense
        layer).
        """
        self._shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten()(x)
        z_mean = Dense(latent_dim, name="z_mean")(x)
        z_log_var = Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        
        return z_mean, z_log_var, z




"""
## Define the VAE as a `Model` with a custom `train_step`
"""





class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        self.reconstruction_loss_weight = 10000.
        self.mse_loss_fn = tf.keras.losses.MeanSquaredError()


    def call(self, inputs, states=None, return_state=False, training=False):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * K.sum(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1.)
        
        mse_loss = self.mse_loss_fn(inputs, reconstructed)
        total_loss =  kl_loss + mse_loss  * self.reconstruction_loss_weight
        self.add_loss(total_loss)


        # self.add_metric(total_loss, name="total_loss")
        self.add_metric(mse_loss, name="mse_loss")
        self.add_metric(kl_loss, name="kl_loss")
        return reconstructed

 
    

"""www
## Train the VAE
"""

def _test_output(number=1):
    reconstructed, latenent_rep = reconstruct(vae, x_train[number][np.newaxis, ...])
    graph(np.squeeze(x_train[number]),np.squeeze(reconstructed), np.squeeze(latenent_rep))
    mse = tf.keras.losses.MeanSquaredError()
    print(mse(reconstructed, x_train[number][np.newaxis, ...]))
    

def VAE_load(latent_space=latent_dim):
    layer_model = VAE_layers(
        input_shape=(FRAME_WIDTH, FRAME_LENGTH, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(1, 2, 2, 2, (2, 1)),
        latent_space_dim=latent_space
    )
    layer_model.encoder.summary()
    layer_model.decoder.summary()
    return layer_model.encoder, layer_model.decoder


x_train = data_out(SAMPLES)
encoder, decoder = VAE_load()
vae = VAE(encoder, decoder)

# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# tb_callback = tf.keras.callbacks.TensorBoard(log_dir)


vae.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0005),
            # metrics=['MeanSquaredError', 'KLDivergence'],
            # callbacks=[tensorboard_callback]
            )

# tb_callback.set_model(vae)
vae.fit(x_train, epochs=EPOCHS, batch_size=128)

"""
## Display a grid of sampled digits
"""
_test_output(1)    





vae.compute_output_shape(input_shape=[128,96,32,1])

save_model(vae, "VAE_model", "tf")


# vae1 = load_model("VAE_model")
# reconstructed, latenent_rep = reconstruct(vae1, x_train[10][np.newaxis, ...])
# graph(np.squeeze(x_train[1]),np.squeeze(reconstructed), np.squeeze(latenent_rep))
# vae1.fit(x_train, epochs=EPOCHS, batch_size=64)


