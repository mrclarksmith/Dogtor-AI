# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 18:11:08 2022

@author: server
"""

"file to preprocess data, and feed the model"

import librosa
import numpy as np
import pickle
import random
import pandas as pd



pickle_path = "D:/python2/woof_friend/pickle_list_MEL_100_NFTT_512"
pickle_path2 = "D:/python2/woof_friend/pickle_list_Newlin_Dog_barks"
pickle_path3 =  "D:/python2/woof_friend/pickle_list_PetDog_Sound_event"
FRAME_LENGTH = 32
FRAME_MEL= 96

class PreprocessData:
    def __init__(self,
                path = [pickle_path, pickle_path2, pickle_path3],
                frame_length=FRAME_LENGTH,
                frame_mel =FRAME_MEL,
                normalize = [0,1],
                ):
        
        self.path = path
        self.frame_length = frame_length
        self.frame_mel = frame_mel
        self.normalize_range = normalize
        self.data = None
        
    def process(self):
        self.data = self.preprocess(self.path)
        return self.data
        

    def preprocess(self,paths):
        self.mfcc_combined = None
        for pickle_path in paths:
            if self.mfcc_combined is None:
                with open(pickle_path, "rb") as f:
                    self.mfcc_combined = pd.DataFrame(pickle.load(f)).T
            else:
                with open(pickle_path, "rb") as f:
                    self.mfcc_combined = pd.concat((self.mfcc_combined, pd.DataFrame(pickle.load(f)).T), axis=0)
                

        # mfcc_combined = pd.concat([mfcc_list.T, mfcc_list2.T, mfcc_list3.T], axis = 0)
        self.mfcc_combined[1] = self.mfcc_combined[1].astype(int)
        self.mfcc_combined = self.mfcc_combined.loc[self.mfcc_combined[1]==3]
        X = list(self.mfcc_combined[0])
            
        for i, item in enumerate(X):
            item = self.power_to_db(item)
            X[i] = self.normalize(item)
    
        
        X_train = self.data_set(X)
        X = np.array(X_train)    
        return X 
    
    def normalize(self, x):
        a = self.normalize_range[0]
        b = self.normalize_range[1]
        x = ( (b-a)*( x- x.min()) / ( x.max() - x.min() ) ) +(a)
        return x
    
    @staticmethod
    def power_to_db(S):
        S_DB = librosa.power_to_db(S, ref=np.max)
        return S_DB
    
    def data_set(self, X):
        X_1 = []
        #Pad the data
        for row, item in enumerate(X):
            if item.shape[1] <self.frame_length:
                
                pad = int( (self.frame_length-item.shape[1])*random.random() )
                           
                X_1.append(
                        np.pad(
                            item,(
                                (0,0),(pad,(self.frame_length-item.shape[1]-pad))
                                ),'constant',constant_values = (0)
                            )
                        )
                
    
            elif item.shape[1] >self.frame_length:
                before = int((item.shape[1]-self.frame_length)*random.random()*.3 )
                X_1.append( item[:,before:self.frame_length+before] )
    
                
                X_1.append( item[:,-(self.frame_length+before+1):-before-1] )
            else:
                X_1.append(item)
                
        
        X_1 = np.dstack( X_1) #convert list to np array
        X_1 = np.rollaxis( X_1,-1)        #bring last axis to front
        X_1 = np.expand_dims( X_1,-1) # add channel of 1 for conv2d to work
        X_1 = X_1[:,(100-self.frame_mel):,::]
        return X_1
    
    

if __name__=="__main__":
    pross =  PreprocessData()
    X = pross.process()