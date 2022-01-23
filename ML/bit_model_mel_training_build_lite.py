# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 16:00:11 2022

@author: Server
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


def normalize(x, a=0, b= 1):
    y = ( (b-a)*( x- x.min()) / ( x.max() - x.min() ) ) +(a)
    return y
def plot_mel(S):
    if len(X_train[1].shape) == 3:
        S= np.squeeze(S)
    sr = 22050
    N_FFT = 512 
    # WIN_LENGTH = N_FFT
    HOP_LENGTH = N_FFT//1
    # S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S, sr=sr,  x_axis='time', y_axis='mel');
    plt.colorbar(format='%+1.0f dB');    
    
def power_to_db(S):
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB




frame_length =97 #176
frame_height = 100

inChannel = 1
x, y = frame_height, frame_length
input_img = Input(shape = (x, y, inChannel))
num_classes = 1    
input_shape = [frame_height,frame_length] #input size
latenet_dim = frame_length

model_path = os.path.join("D:/python2/woof_friend/bit_m-r101x1_1")
pickle_path = "D:/python2/woof_friend/pickle_list_MEL_100_NFTT_512"
pickle_path2 = "D:/python2/woof_friend/pickle_list_Newlin_Dog_barks"
module = hub.KerasLayer(model_path)
# module = tf.saved_model.load(model_path)
def initialize(frame_length = frame_length ):
    with open(pickle_path, "rb") as f:
        mfcc_list = pickle.load(f)
    with open(pickle_path2, "rb") as f:
        mfcc_list2 = (pickle.load(f))        
    
    for i in range(len(mfcc_list)):   
        mfcc_list[i].extend(mfcc_list2[i])        
        
        
    X = mfcc_list[0]
    for i, item in enumerate(X):
        item = power_to_db(item)
        X[i] = normalize(item)
    
    y = np.array(mfcc_list[1])
    y = np.array(y, dtype=int)
    y = np.where(y == 3, 1, 0)
    return X , y
    

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
    X_1 = np.rollaxis( X_1,-1)        #bring last axis to front
    X_1 = np.expand_dims( X_1,-1) # add channel of 1 for conv2d to work

       
    # X_train, X_test,  y_train, y_test  =  train_test_split(X,y, test_size= .15, random_state=42) 
    
    y_1  = np.array(y_1)

    # y_train  = to_categorical(y_train)
    # y_test  = to_categorical(y_test)
    return  X_1, y_1



# def block_horiz(z):
#     r = random.random() 
#     n = int(1/(r+0.42)**4 +1)
#     e = int(frame_length/n/n*r*10)
#     for l in range(n):
#         s = int( ((frame_length+(l*155*r*7) ) ) %frame_length )
#         w = int( ((frame_length+(l*588*r*11) ) %n*8 ) % e*3 )
#         print(w,"w", s, "s", n, "n")
#         z[:,s:s+w]= 0
#     return z        
            

def block_horiz(z):
    r = random.random()
    r2 = int(random.random()*2.6+.5)
    
    for l in range(1,r2+1):
        s = int( ( frame_length*r*l**r2*33) %frame_length)
        w = int(( r2*r*1/(l+5)*5+1)*2 ) 
        # print(w,"w", s, "s")
        z[:,s:s+w]= 0
    return z        
            

def block_vert(z):
    r = random.random()
    r2 = int(random.random()*2.6+.5)
    
    for l in range(1,r2+1):
        s = int( ( frame_height*r*l**r2*33) %frame_height)
        w = int(( r2*r*1/(l+5)*5+1)*2 ) 
        # print(w,"w", s, "s")
        z[s:s+w,]= 0
    return z        
            




# def data_set_prep(item):
#     global item1
#     item1 = item
#     print(item1)
#     #Pad the data
#     #TODO make into a function to use also in data_set func
#     if  item.to_tensor()) <frame_length:
#         item = np.pad(item,((0,0),(0,frame_length-item.shape[1])),'constant',constant_values = (0))
#     elif item.shape[1] >frame_length:
#         before = int((item.shape[1]-frame_length)*random.random() )
#         item = item[:,before:frame_length+before]

#     # item = np.dstack(item) #convert list to np array
#     # item = tf.roll(item,-1)        #bring last axis to front
#     item = tf.expand_dims(item,-1) # add channel of 1 for conv2d to work

       
#     # X_train, X_test,  y_train, y_test  =  train_test_split(X,y, test_size= .15, random_state=42) 
    
    
#     # y_train  = to_categorical(y_train)
#     # y_test  = to_categorical(y_test)
#     return item

X,y=initialize()
X_tr, X_test, y_tr, y_test = train_test_split(X,y, test_size= .15) 

X_train, y_train = data_set(X_tr, y_tr)
X_test, y_test = data_set(X_test, y_test)

# dataset =  tf.data.Dataset.from_tensor_slices(X_train)
# dataset =  dataset.map(data_set_prep, dataset)

# dataset = dataset.batch(
#         batch_size).prefetch(
#             tf.data.AUTOTUNE)


# def preprocess(item, label):
#     if item.shape[1] <frame_length:
#         item = np.pad(item,((0,0),(0,frame_length-item.shape[1])),'constant',constant_values = (0))
#     elif item.shape[1] >frame_length: #crop item
#         before = (item.shape[1]-frame_length)*random.random()
#         # after  = item.shape[1]-frame_length- before
#         item = item[:,before:before+frame_length]
#     return item, label

# dataset = tf.data.Dataset.from_generator(data_set_prep,
#                                          args = (X_train),
#                                          output_signature=(tf.TensorSpec(shape=(100,frame_length), dtype=tf.float32),
#                                                            tf.TensorSpec(shape=(), dtype=tf.int32))
#                                                                             )



# dataset =  tf.data.Dataset.from_tensor_slices((X_train, y_train))
# dataset =  (dataset
#             .shuffle(500)
#             .map(preprocess)
#             )



# iterator = dataset.make_initializable_iterator()




class New_model(tf.keras.Model):
    def __init__(self, module):
        super().__init__()
        # self.num_classes =  num_classes
        self.pre2d =tf.keras.layers.Conv2D(3, 2, strides=(1, 1), padding='same', activation='swish', name= "pre2d")
        # tensor = tf.div(tf.subtract(tensor, tf.reduce_min(tensor) ), 
        #                 tf.subtract(tf.reduce_max(tensor),tf.reduce_min(tensor)
        #                 )
        self.dense1 = tf.keras.layers.Dense(512, activation = 'relu', name = 'dense1')
        self.dense2 = tf.keras.layers.Dense(256, activation = 'relu', name = 'dense2')
        self.head =  tf.keras.layers.Dense(1, activation ='sigmoid')
        
        self.bit_model =  module
        
    def call(self, images):
        pre2d = self.pre2d(images)
        bit_model =  self.bit_model(pre2d)
        bit_embedding =  self.dense1(bit_model) 
        bit_embedding =  self.dense2(bit_embedding) 
        return self.head(bit_embedding)
    
    def model(self):
        inputs = tf.keras.layers.Input(shape = (100,frame_length,1))
        outputs = self.call(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="model")
    
    


# def cast_to_tuple(features):
#   return (features['image'], features['label'])
  
# def preprocess_train(images):
#   # Apply random crops and horizontal flips for all tasks 
#   # except those for which cropping or flipping destroys the label semantics
#   # (e.g. predict orientation of an object)
#   images = tfa.image.sparse_image_warp(images)
#   images = tfio.audio.freq_mask(dbscale_mel_spectrogram, param=10)
#   images = tfio.audio.time_mask(dbscale_mel_spectrogram, param=10)
#   return features



# pipeline_train = (ds_train
#                   .shuffle(10000)
#                   .repeat(int(SCHEDULE_LENGTH * BATCH_SIZE / DATASET_NUM_TRAIN_EXAMPLES * STEPS_PER_EPOCH) + 1 + 50)  # repeat dataset_size / num_steps
#                   .map(preprocess_train, num_parallel_calls=8)
#                   .batch(BATCH_SIZE)
#                     # for keras model.fit
#                   .prefetch(2))

# pipeline_test = (ds_test.batch(BATCH_SIZE).prefetch(2))



model = New_model( module = module)
SCHEDULE_LENGTH = 45
SCHEDULE_BOUNDARIES = [200, 300, 400, 500]
# SCHEDULE_LENGTH = SCHEDULE_LENGTH * 512 / BATCH_SIZE        
BATCH_SIZE = 64
lr = 0.003 * BATCH_SIZE / 512 
STEPS_PER_EPOCH = 8
EPOCHS = 20

# Define optimiser and loss
# Decay learning rate by a factor of 10 at SCHEDULE_BOUNDARIES.
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=SCHEDULE_BOUNDARIES, 
                                                                   values=[lr, lr*0.1, lr*0.1, lr*0.001, lr*0.0001])
# lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=SCHEDULE_BOUNDARIES,   values=[lr*0.001, lr*0.0001, lr*0.00001, lr*0.00001])
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
# optimizer = tfa.optimizers.RectifiedAdam() # cannot make model with TFA into a light model will fail every time 
# optimizer = tf.keras.optimizers.Adam()

loss_fn = tf.keras.losses.BinaryCrossentropy()


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='E:/python/1c_f_model.h5',
        save_weights_only=True,
        monitor='accuracy',
        mode='auto',
        save_freq=BATCH_SIZE,
        save_best_only=True,
        )
             
        
model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=['accuracy'])

def load_check():
    model.build(input_shape = (BATCH_SIZE,100,frame_length,1))
    model.load_weights('E:/python/1c_f_model.h5')

def chain_train(carts):
    global X_train
    global y_train
    global X_tr
    global y_tr
    
    for i in range(carts):
        print(f"chain_train {i}")
        X_train, y_train = data_set(X_tr, y_tr, train=True)
        
        train()
        


def train(X_train =X_train, y_train=y_train):
# Fine-tune model
    history = model.fit(X_train,y_train,
        batch_size=BATCH_SIZE,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs= SCHEDULE_LENGTH ,  # TODO: replace with `epochs=10` here to shorten fine-tuning for tutorial if you wish
        validation_data=[X_test,y_test],
        callbacks = [model_checkpoint_callback],
        )
    return history
def save_model(model):
    import time
    ts = int(time.time())
    file_path =  f"D:/python2/woof_friend/models/woof_detector/Dogtor-AI\models\woof_detector\{ts}"
    model.save(filepath=file_path, save_format='tf')


def load_full_model():
    model = tf.keras.models.load_model(r'D:\python2\woof_friend\Dogtor-AI\models\woof_detector\1641358780')
    return model
def save_lite():
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    tflite_model = converter.convert()
    # Convert the model
    # converter = tf.lite.TFLiteConverter.from_saved_model('D:\python2\woof_friend\Dogtor-AI\models\woof_detector/1641358780') # path to the SavedModel directory
    # tflite_model = converter.convert()
    # interpreter  = tf.lite.Interpreter(model_path = 'woof_friend_final.tflite')
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    # Save the model.
    with open('woof_friend_final1.tflite', 'wb') as f:
      f.write(tflite_model)

    
     

