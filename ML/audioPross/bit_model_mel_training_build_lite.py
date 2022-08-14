# -*- coding: utf-8 -*-
'''Written by Alexsey Gromov
"Model used https://tfhub.dev/google/collections/bit"
'''
"""
Needed for DogPI
"""


#from sklearn.metrics import accuracy_score, precision_score, recall_score


import _audio_helper as ah  # custom library ah=AudioHelper
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
import time
tf.random.set_seed(42)


tf.get_logger().setLevel("ERROR")

FRAME_LENGTH = 97  # NUMBER OF TIME SAMPLES
FRAME_HEIGHT = 96  # Number of mel channels
sr = 22050
N_FFT = 512
HOP_LENGTH = N_FFT//2
# Defining hyperparameters
DESIRED_SAMPLES = FRAME_LENGTH*HOP_LENGTH  # 24,832 samples

BATCH_SIZE = 8

# makes sure not to delete this folder :)
model_path = os.path.join("D:/python2/woof_friend/bit_m-r101x1_1")
module = hub.KerasLayer(model_path)
# module = tf.saved_model.load(model_path)


class New_model(tf.keras.Model):
    def __init__(self, module, frame_height, frame_length):
        super().__init__()
        self.pre2d = tf.keras.layers.Conv2D(3, (3, 3), padding="same", name='pre2d')
        self.dense1 = tf.keras.layers.Dense(512, activation='relu', name='dense1')
        self.dense2 = tf.keras.layers.Dense(256, activation='relu', name='dense2')
        self.head = tf.keras.layers.Dense(1, activation='sigmoid')
        self.frame_length = frame_length
        self.frame_height = frame_height
        self.bit_model = module

    def call(self, images):
        #pre2d = tf.keras.layers.concatenate([images, images, images], axis=3, name="pre2d_conc")
        pre2d = self.pre2d(images)
        bit_model = self.bit_model(pre2d)
        bit_embedding = self.dense1(bit_model)
        bit_embedding = self.dense2(bit_embedding)
        return self.head(bit_embedding)

    def model(self):
        inputs = tf.keras.layers.Input(shape=(self.frame_height, self.frame_length, 1))
        outputs = self.call(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="model")

    # def train_step(self, batch):
    #     x_batch_train, y_batch_train = batch
    #     with tf.GradientTape() as tape:
    #         logits = self.model(x_batch_train, training=True)
    #         loss_value = loss_fn(y_batch_train)

# Needed to use parallel processing of a class


class bit_trainer:
    def __init__(self, model, dog_dir, not_dog_dir, frame_length, frame_height, batch_size, steps_per_epoch, epochs, test_number_of_files=None):
        self.model = model
        self.dog_dir = dog_dir
        self.not_dog_dir = not_dog_dir
        self.frame_length = frame_length
        self.frame_height = frame_height
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.test_number_of_files = test_number_of_files  # Test program on a small slice of a dataset
        self.initialize_files()

    def initialize_files(self):
        # Initialize the location of the wave files
        self.dog_files = self._get_wav_files(self.dog_dir)
        self.not_dog_files = self._get_wav_files(self.not_dog_dir)

    def initialize_data(self):
        global dog_pd, not_dog_pd, array1, array2, e1, e2

        bit_model.dog_mel = self.process_data(bit_model.dog_files[:self.test_number_of_files])
        bit_model.not_dog_mel = self.process_data(
            self.not_dog_files[:self.test_number_of_files])

        dog_pd = pd.DataFrame({'X': self.dog_mel, 'y': np.ones(len(self.dog_mel))})
        not_dog_pd = pd.DataFrame({'X': self.not_dog_mel, 'y': np.zeros(len(self.not_dog_mel))})

        # split data into test and train, based on dog and not dog, that way you can combine multiple dog barks into 2 for training variation later

        self.data = pd.concat([dog_pd, not_dog_pd], ignore_index=True)
        X_train, X_test, y_train, y_test = self.split_data(self.data.X, self.data.y, test_size=.18)

        #self.X_train_list = X_train.tolist()
        self.data_train = list(zip(X_train, y_train))

        #self.X_train = np.array(X_train.tolist())[...,  np.newaxis]
        self.X_test = np.array(X_test.tolist())[...,  np.newaxis]
        #self.y_train = np.array(y_train.tolist())[...,  np.newaxis]
        self.y_test = np.array(y_test.tolist())[...,  np.newaxis]

        self.compile_model()

    @staticmethod
    def _get_wav_files(directory):
        fu = [os.path.join(dp, f) for dp, dn, filenames in os.walk(directory) for f in filenames if
              os.path.splitext(f)[1].lower() == '.wav']
        return fu

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_loss_fn(self, loss_fn):
        self.loss = loss_fn

    def compile_model(self, metrics="accuracy"):
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_fn,
                           metrics=[metrics])

    def checkpoint_data(self):
        self.callback_save_dir = "E:/python/1c_f_model_test.h5"
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.callback_save_dir,
            save_weights_only=True,
            monitor='accuracy',
            mode='auto',
            save_freq=self.batch_size,
            save_best_only=True,
        )

        self.model_checkpoint_callback = model_checkpoint_callback

    def train(self):
        for epoch in range(self.epochs):
            print(epoch, " out of ", self.epochs)
            X_train_agmented, y_train = self.prepross_epoch(
                self.data_train)  # Squeeze not needed in this code
            X_train_agmented = np.array(X_train_agmented)[...,  np.newaxis]
            y_train = np.array(y_train)[...,  np.newaxis]
            # add scheduler for learning rate as it resets each epoch with model.fit function
            self.model.fit(X_train_agmented, y_train,
                           batch_size=self.batch_size,
                           steps_per_epoch=self.steps_per_epoch,
                           epochs=3,
                           validation_data=[self.X_test, self.y_test],
                           callbacks=[self.model_checkpoint_callback],
                           )

    def load_check(self, callback_save_dir):
        self.callback_save_dir = callback_save_dir
        self.model.build(input_shape=(self.batch_size, self.frame_height, self.frame_length, 1))
        self.model.load_weights(callback_save_dir)

    @staticmethod
    def _normalize(x, a=0, b=1):
        x = ((b-a)*(x - x.min()) / (x.max() - x.min())) + (a)
        return x

    @staticmethod
    def _power_to_db(S):
        S_DB = librosa.power_to_db(S, ref=np.max)
        return S_DB


# randomly pad data in the front


    @staticmethod
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

    @staticmethod
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

    def preprocess(self, filename, desired_samples=DESIRED_SAMPLES, load=True):
        frame_height = self.frame_height  # NUMBER OF MELS
        FMIN = 200
        if load:
            audio, _ = librosa.load(filename, res_type='kaiser_fast', sr=sr, mono=True)
            audio = np.trim_zeros(audio)
            audio = audio[:desired_samples]
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
        mel = self._power_to_db(mel)
        mel = self._normalize(mel)
        # NUMBER OF TIME SAMPLES
        mel = self._data_pad(mel, self.frame_length)
        return mel

    @staticmethod
    def remove_leading_trailing_zeros(t):
        p = np.where(t != 0)
        t = t[:, min(p[1]): max(p[1]) + 1]
        #t = t[min(p[0]): max(p[0]) + 1, min(p[1]): max(p[1]) + 1]
        return t

    # This step happens on each epoch
    def prepross_epoch(self, mels):
        new_mels = []
        y_train = []
        for train_data_point in mels:
            mel = train_data_point[0]
            y = train_data_point[1]
            mel = self.remove_leading_trailing_zeros(mel)
            # Randomly concatonate 2 different barks
            if (random.random() < .5) & (mel.shape[1] < 80) & (y == 1):
                mel_add = self.remove_leading_trailing_zeros(random.choice(self.dog_mel))
                mel = np.concatenate((mel, mel_add), axis=1)
            # Add random blockout blocks to obfuscate the data

            mel = self._data_pad(mel, self.frame_length)
            if random.random() < .7:
                mel = self.block_horiz(mel)
            if random.random() < .2:
                # not sure if this helps due to nature of dog bark being a vertical activation of frequencies
                mel = self.block_vert(mel)
            new_mels.append(mel)
            y_train.append(y)
        return new_mels, y_train

    def process_data(self, mylist):
        results = []  # Mel frequency
        num_items = len(mylist)
        print(num_items)
        print("generating mels")
        for i, item in enumerate(mylist):
            if (i % 200) == 0:
                print(i, end=" ")
            results.append(self.preprocess(item))
        return results

    @staticmethod
    def split_data(X, y, test_size=0.20, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def block_horiz(self, z):
        r = random.random()
        r2 = int(random.random()*2.6+.5)

        for l in range(1, r2+1):
            s = int((self.frame_length*r*l**r2*33) % self.frame_length)
            w = int((r2*r*1/(l+5)*5+1)*2)
            z[:, s:s+w] = 0
        return z

    def block_vert(self, z):
        r = random.random()
        r2 = int(random.random()*2.6+.5)

        for l in range(1, r2+1):
            s = int((self.frame_height*r*l**r2*33) % self.frame_height)
            w = int((r2*r*1/(l+5)*5+1)*2)
            # print(w,"w", s, "s")
            z[s:s+w, ] = 0
        return z

    @staticmethod
    def normalize(x, a=0, b=1):
        y = ((b-a)*(x - x.min()) / (x.max() - x.min())) + (a)
        return y

    def plot_mel(self, S):
        if len(self.X_test[1].shape) == 3:
            S = np.squeeze(S)
        sr = 22050
        librosa.display.specshow(S, sr=sr,  x_axis='time', y_axis='mel')
        plt.colorbar(format='%+1.0f dB')

    def _epoch_plot_training_data(self, data):
        plt.figure(figsize=(10, 10))
        for i in range(9):
            img = random.choice(data)
            ax = plt.subplot(3, 3, i + 1)
            self.plot_mel(img)

    @staticmethod
    def power_to_db(S):
        S_DB = librosa.power_to_db(S, ref=np.max)
        return S_DB

    def save_model(self, time_bool=True, suffix=""):
        if time_bool:
            ts = int(time.time())
        else:
            ts = ""

        self.model_save_file_path = f"D:/python2/woof_friend/models/woof_detector/Dogtor-AI\models\woof_detector\model_{suffix}{ts}"
        self.model.save(filepath=self.model_save_file_path, save_format='tf')

    def load_full_model(self, load_latest=False):
        if load_latest:
            try:
                self._load_model(self.model_save_file_path)
            except Exception as e:
                print(e)
                print("save path not loaded or not initialized")
                file = ah.select_file()
                self._load_model(file)
        else:
            file = ah.select_file()
            self._load_model(file)

    def _load_model(self, file):
        self.model = tf.keras.models.load_model(file)

    def save_lite(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter._experimental_lower_tensor_list_ops = False

        tflite_model = converter.convert()
        print("text")
        # Save the model.
        with open('woof_friend_final.tflite', 'wb') as f:
            f.write(tflite_model)
        print("saved as woof_friend_final.tflite")

###################################################################################################


def _unit_test(bit_model):
    # main()
    bit_model.dog_files = bit_model.dog_files[:50]
    bit_model.not_dog_files = bit_model.not_dog_files[:50]
    return bit_model


dog_dir = "D:/python2/woof_friend/Dogtor-AI/ML/data/ML_SET/dog"
not_dog_dir = "D:/python2/woof_friend/Dogtor-AI/ML/data/ML_SET/not_dog"

model = New_model(module=module, frame_height=FRAME_HEIGHT, frame_length=FRAME_LENGTH)

SCHEDULE_LENGTH = 25
SCHEDULE_BOUNDARIES = [200, 300, 400, 500]
# SCHEDULE_LENGTH = SCHEDULE_LENGTH * 512 / BATCH_SIZE
BATCH_SIZE = 64
lr = 0.003 * BATCH_SIZE / 512
STEPS_PER_EPOCH = 8
EPOCHS = 22


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005,)
loss_fn = tf.keras.losses.BinaryCrossentropy()

bit_model = bit_trainer(model, dog_dir, not_dog_dir, FRAME_LENGTH,
                        FRAME_HEIGHT, BATCH_SIZE, STEPS_PER_EPOCH, EPOCHS, test_number_of_files=None)  # None takes all of the data.

bit_model.loss_fn = loss_fn
bit_model.optimizer = optimizer

# Process waves files  dog, not dog


bit_model.initialize_data()
bit_model.checkpoint_data()

bit_model.train()
bit_model.save_lite()  # saves woof_friend_final.tflite


def check_lite_model():
    interpreter = tf.lite.Interpreter("woof_friend_final.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    output_data_list = []
    for data in bit_model.X_test:
        input_data = data[np.newaxis, ...]
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data_list.append(output_data[0][0])
        print(output_data)
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.

    # print(output_data)
    return output_data_list


def generate_stft(audios):
    generated_stfts = []
    for audio in audios:
        stft = librosa.stft(np.squeeze(audio), hop_length=256, win_length=512)
        generated_stfts.append(stft)

    return generated_stfts


if __name__ == '__main__':
    pass
