import os
import librosa
import numpy as np
import pickle
from autoencoder import VAE
import random
import matplotlib.pyplot as plt
import pandas as pd



LEARNING_RATE = 0.0005
BATCH_SIZE = 64
EPOCHS = 150

pickle_path = "D:/python2/woof_friend/pickle_list_MEL_100_NFTT_512"
pickle_path2 = "D:/python2/woof_friend/pickle_list_Newlin_Dog_barks"
pickle_path3 =  "D:/python2/woof_friend/pickle_list_PetDog_Sound_event"
FRAME_LENGTH = 32
FRAME_WIDTH= 96

def preprocess(pickle_path, pickle_path2, pickle_path3, FRAME_LENGTH):
    with open(pickle_path, "rb") as f:
        mfcc_list = pd.DataFrame(pickle.load(f))
    with open(pickle_path2, "rb") as f:
        mfcc_list2 = pd.DataFrame(pickle.load(f))
    with open(pickle_path3, "rb") as f:
        mfcc_list3 = pd.DataFrame(pickle.load(f))           
    
    
    mfcc_combined = pd.concat([mfcc_list.T, mfcc_list2.T, mfcc_list3.T], axis = 0)
    mfcc_combined[1] = mfcc_combined[1].astype(int)
    mfcc_combined = mfcc_combined.loc[mfcc_combined[1]==3]
    X = list(mfcc_combined[0])
        
    for i, item in enumerate(X):
        item = power_to_db(item)
        X[i] = normalize(item)

    
    X_train = data_set(X, train=True)
    X = np.array(X_train)    
    return X 

def normalize(x, a=0, b= 1):
    x = ( (b-a)*( x- x.min()) / ( x.max() - x.min() ) ) +(a)
    return x

def power_to_db(S):
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB


def data_set(X,train = False):
    X_1 = []
    #Pad the data
    for row, item in enumerate(X):
        if item.shape[1] <FRAME_LENGTH:
            
            pad = int( (FRAME_LENGTH-item.shape[1])*random.random() )
                       
            X_1.append(
                    np.pad(
                        item,(
                            (0,0),(pad,(FRAME_LENGTH-item.shape[1]-pad))
                            ),'constant',constant_values = (0)
                        )
                    )
            

        elif item.shape[1] >FRAME_LENGTH:
            before = int((item.shape[1]-FRAME_LENGTH)*random.random()*.3 )
            X_1.append( item[:,before:FRAME_LENGTH+before] )

            
            X_1.append( item[:,-(FRAME_LENGTH+before+1):-before-1] )
        else:
            X_1.append(item)
            
    
    X_1 = np.dstack( X_1) #convert list to np array
    X_1 = np.rollaxis( X_1,-1)        #bring last axis to front
    X_1 = np.expand_dims( X_1,-1) # add channel of 1 for conv2d to work
    X_1 = X_1[:,(100-FRAME_WIDTH):,::]
    return  X_1

def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = VAE(
        input_shape=(FRAME_WIDTH, FRAME_LENGTH, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(1, 2, 2, 2, (2, 1)),
        latent_space_dim=128
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)
    return autoencoder


if __name__ == "__main__":
    
    # x_train = preprocess(pickle_path, pickle_path2, pickle_path3, FRAME_LENGTH )
    
    # autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    # autoencoder.save("model")
    autoencoder = VAE.load("model")
