# -*- coding: utf-8 -*-
'''Written by Alexsey Gromov

'''
"""
Needed for DogPI
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle 
import random
import librosa
import librosa.display
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Flatten, Dense, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, Dropout, Reshape
from tensorflow.keras.utils import to_categorical
import tensorflow_hub as hub
import tensorflow_addons as tfa
import time
import _audio_helper as ah #custom library ah=AudioHelper
tf.random.set_seed(42)


# Setting logger level to avoid input shape warnings
tf.get_logger().setLevel("ERROR")

FRAME_LENGTH = 97 #NUMBER OF TIME SAMPLES
FRAME_HEIGHT = 96 #Number of mel channels
sr = 22050
N_FFT = 512
HOP_LENGTH = N_FFT//2
# Defining hyperparameters
DESIRED_SAMPLES = FRAME_LENGTH*HOP_LENGTH # 24,832 samples

LEARNING_RATE_GEN = 1e-5
LEARNING_RATE_DISC = 1e-5
BATCH_SIZE = 32


# frame_height = 100

# inChannel = 1
# x, y = frame_height, frame_length
# input_img = Input(shape = (x, y, inChannel))
# num_classes = 1    
# input_shape = [frame_height,frame_length] #input size
# latenet_dim = frame_length

model_path = os.path.join("D:/python2/woof_friend/bit_m-r101x1_1") # makes sure not to delete this folder :)

module = hub.KerasLayer(model_path)
# module = tf.saved_model.load(model_path)


class New_model(tf.keras.Model):
    def __init__(self, module, frame_height, frame_length):
        super().__init__()
        self.pre2d =tf.keras.layers.Conv2D(3,(3,3),padding='same', name= "pre2d")
        self.dense1 = tf.keras.layers.Dense(512, activation = 'relu', name = 'dense1')
        self.dense2 = tf.keras.layers.Dense(256, activation = 'relu', name = 'dense2')
        self.head =  tf.keras.layers.Dense(1, activation ='sigmoid')
        self.frame_length = frame_length
        self.frame_height = frame_height
        self.bit_model =  module
        
        
    def call(self, images):
        pre2d = self.pre2d(images)
        bit_model =  self.bit_model(pre2d)
        bit_embedding =  self.dense1(bit_model) 
        bit_embedding =  self.dense2(bit_embedding) 
        return self.head(bit_embedding)
    
    def model(self):
        inputs = tf.keras.layers.Input(shape = (self.frame_height,self.frame_length,1))
        outputs = self.call(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="model")
    


class  bit_trainer:
    def __init__(self, model, dog_dir, not_dog_dir, frame_length, frame_height, batch_size, steps_per_epoch, epochs):
        self.model = model
        self.dog_dir = dog_dir
        self.not_dog_dir = not_dog_dir
        self.frame_length = frame_length
        self.frame_height = frame_height
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        
        self.initialize_files()
        
    def initialize_files(self):
        #Initialize the location of the wave files
        self.dog_files = self._get_wav_files(self.dog_dir)
        self.not_dog_files =  self._get_wav_files(self.not_dog_dir)
        
        
    def initialize_data(self):
        global dog_pd, not_dog_pd, array1, array2, e1, e2
        
        #Process waves files  dog, not dog
        self.dog_mel = self.process_data(self.dog_files)
        self.not_dog_mel = self.process_data(self.not_dog_files)
        
        dog_pd = pd.DataFrame({'X':self.dog_mel, 'y' : np.ones(len(self.dog_mel))})
        not_dog_pd = pd.DataFrame({'X':self.not_dog_mel, 'y' : np.zeros(len(self.not_dog_mel))}) 
        
        self.data = pd.concat([dog_pd, not_dog_pd],ignore_index=True)    
        X_train, X_test, y_train, y_test = self.split_data(self.data.X, self.data.y, test_size=.18)

        self.X_train = np.array(X_train.tolist())[...,  np.newaxis]
        self.X_test = np.array(X_test.tolist())[...,  np.newaxis]
        self.y_train = np.array(y_train.tolist())[...,  np.newaxis]
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
        # Fine-tune model
        self.history = self.model.fit(self.X_train, self.y_train,
            batch_size=self.batch_size,
            steps_per_epoch=self.steps_per_epoch,
            epochs = self.epochs,  # TODO: replace with `epochs=10` here to shorten fine-tuning for tutorial if you wish
            validation_data=[self.X_test, self.y_test],
            callbacks = [self.model_checkpoint_callback],
            )
        
    def load_check(self, callback_save_dir):
        self.callback_save_dir = callback_save_dir
        self.model.build(input_shape = (self.batch_size,self.frame_height,self.frame_length,1))
        self.model.load_weights(callback_save_dir)
        
    @staticmethod       
    def _normalize(x, a=0, b= 1):
        x = ( (b-a)*( x- x.min()) / ( x.max() - x.min() ) ) +(a)
        return x
    @staticmethod
    def _power_to_db(S):
        S_DB = librosa.power_to_db(S, ref=np.max)
        return S_DB
    @staticmethod
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
    @staticmethod
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
    
    def preprocess(self, filename, desired_samples=DESIRED_SAMPLES, load=True):


        # n_mfcc = FRAME_HEIGHT              
        # window_type = 'hann' 
        # feature = 'mel'     
        frame_lenght= self.frame_length #NUMBER OF TIME SAMPLES
        frame_height = self.frame_height #NUMBER OF MELS
        FMIN = 200
        if load:        
            audio , _ = librosa.load(filename, res_type='kaiser_fast', sr=sr, mono=True)
            audio  = np.trim_zeros(audio)
            audio = audio[:desired_samples] 
            # audio = np.pad(audio, (0, desired_samples-audio.size) , mode='constant')
        else:
            audio = filename
        # audio = librosa.util.normalize(audio)
        mel = librosa.feature.melspectrogram(audio,sr, 
                                             n_fft=N_FFT,
                                             hop_length=HOP_LENGTH, 
                                             n_mels=frame_height, 
                                             htk=False, 
                                             fmin=FMIN
                                            )
    
        # Taking the magnitude of the STFT output  
        mel = self._power_to_db(mel)
        mel = self._normalize(mel)
    
        mel = self._data_pad(mel, frame_lenght)
        # Add random blockout blocks to obfuscate the data
        mel = self.block_horiz(mel)
        mel = self.block_vert(mel)
        return mel
    
    
    def process_data(self, mylist):
        result1 = [] # Mel frequency
        num_items =  len(mylist)
        print("generating mels", num_items)
        for i, item in enumerate(mylist):
            if num_items % 200 == 0:
                print(i)
            r1 = self.preprocess(item)
            result1.append(r1)
        return result1
    
    @staticmethod
    def split_data(X, y, test_size=0.20, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
    
    def save_model(self, time_bool=True, suffix=""):
        if time_bool:
            ts = int(time.time())
        else: 
            ts = ""
            
        self.model_save_file_path =  f"D:/python2/woof_friend/models/woof_detector/Dogtor-AI\models\woof_detector\model_{suffix}{ts}"
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
        
    def _load_model(self,file):
        self.model = tf.keras.models.load_model(file)

    def save_lite(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        tflite_model = converter.convert()
        # Convert the model
        # converter = tf.lite.TFLiteConverter.from_saved_model('D:\python2\woof_friend\Dogtor-AI\models\woof_detector/1641358780') # path to the SavedModel directory
        # tflite_model = converter.convert()
        # interpreter  = tf.lite.Interpreter(model_path = 'woof_friend_final.tflite')
        # input_details = interpreter.get_input_details()
        # output_details = interpreter.get_output_details()
        # Save the model.
        with open('woof_friend_final.tflite', 'wb') as f:
          f.write(tflite_model)
    
    def block_horiz(self,z):
        r = random.random()
        r2 = int(random.random()*2.6+.5)
        
        for l in range(1,r2+1):
            s = int( ( self.frame_length*r*l**r2*33) %self.frame_length)
            w = int(( r2*r*1/(l+5)*5+1)*2 ) 
            # print(w,"w", s, "s")
            z[:,s:s+w]= 0
        return z        
                
    def block_vert(self,z):
        r = random.random()
        r2 = int(random.random()*2.6+.5)
        
        for l in range(1,r2+1):
            s = int( ( self.frame_height*r*l**r2*33) %self.frame_height)
            w = int(( r2*r*1/(l+5)*5+1)*2 ) 
            # print(w,"w", s, "s")
            z[s:s+w,]= 0
        return z     
    @staticmethod
    def normalize(x, a=0, b= 1):
        y = ( (b-a)*( x- x.min()) / ( x.max() - x.min() ) ) +(a)
        return y
    def plot_mel(self,S):
        if len(self.X_train[1].shape) == 3:
            S= np.squeeze(S)
        sr = 22050
        # WIN_LENGTH = N_FFT
        # S_DB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S, sr=sr,  x_axis='time', y_axis='mel');
        plt.colorbar(format='%+1.0f dB');    
    @staticmethod    
    def power_to_db(S):
        S_DB = librosa.power_to_db(S, ref=np.max)
        return S_DB                

    
def _unit_test(bit_model):
    #main()
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
EPOCHS = 100

# Define optimiser and loss
# Decay learning rate by a factor of 10 at SCHEDULE_BOUNDARIES.
# lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=SCHEDULE_BOUNDARIES, 
#                                                                    values=[lr, lr*0.1, lr*0.1, lr*0.001, lr*0.0001])
# lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=SCHEDULE_BOUNDARIES,   values=[lr*0.001, lr*0.0001, lr*0.00001, lr*0.00001])
# optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
# foptimizer = tfa.optimizers.RectifiedAdam() # cannot make model with TFA into a light model will fail every time 




optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.BinaryCrossentropy()
        
bit_model = bit_trainer(model, dog_dir, not_dog_dir, FRAME_LENGTH, FRAME_HEIGHT, BATCH_SIZE, STEPS_PER_EPOCH, EPOCHS)
# bit_model = _unit_test(bit_model)
bit_model.loss_fn = loss_fn
bit_model.optimizer = optimizer
bit_model.initialize_data()
bit_model.checkpoint_data()

bit_model.train()
bit_model.save_lite() # saves woof_friend_final.tflite
    

    
    
def generate_stft(audios):
    generated_stfts = []
    for audio in audios:
        
        stft = librosa.stft(np.squeeze(audio), hop_length=256, win_length=512 )
        generated_stfts.append(stft)
    
    return generated_stfts


# generates empty space on each side randomly at each pass
def data_set(X,y,train = False):
    X_1 = []
    y_1= []
    #Pad the data
    for row, (item, z) in enumerate(zip(X,y)):
        if item.shape[1] <frame_length:
            
            pad = int( (frame_length-item.shape[1])*random.random() )
                       
            X_1.append(
                    np.pad(
                        item,(
                            (0,0),(pad,(frame_length-item.shape[1]-pad))
                            ),'constant',constant_values = (0)
                        )
                    )
            y_1.append(z)
        elif item.shape[1] >frame_length:
            before = int((item.shape[1]-frame_length)*random.random()*.3 )
            X_1.append( item[:,before:frame_length+before] )
            y_1.append(z)
            
            X_1.append( item[:,-(frame_length+before+1):-before-1] )
            y_1.append(z)
            
        else:
            X_1.append(item)
            y_1.append(z)
    
    if train == True:
        for i, new_p in enumerate(X_1):
            t= block_horiz(new_p)
            X_1[i]= block_horiz(t)
            
    
    X_1 = np.dstack( X_1) #convert list to np array
    X_1 = np.rollaxis( X_1,-1) #bring last axis to front
    X_1 = np.expand_dims( X_1,-1) # add channel of 1 for conv2d to work

       
    # X_train, X_test,  y_train, y_test  =  train_test_split(X,y, test_size= .15, random_state=42) 
    
    y_1  = np.array(y_1)

    # y_train  = to_categorical(y_train)
    # y_test  = to_categorical(y_test)
    return  X_1, y_1


