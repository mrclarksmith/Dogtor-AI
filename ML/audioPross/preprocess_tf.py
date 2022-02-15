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
FRAME_WIDTH= 96

class PreprocessData:
    
    def __init__(self,
                path = [pickle_path, pickle_path2, pickle_path3],
                FRAME_LENGTH=FRAME_LENGTH,
                FRAME_WIDTH=FRAME_WIDTH,
                NORMALIZE = [0,1],
                ):
        
        self.path = path
        self.frame_length = FRAME_LENGTH
        self.frame_width = FRAME_WIDTH
        self.normalize_range = NORMALIZE
        self.data = None
        
    def process(self):
        self.data = self.preprocess(self.path, self.frame_length)
        return self.data
        

    def preprocess(self,paths, FRAME_LENGTH):
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
    
    @staticmethod
    def data_set(X):
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
        return X_1
    
    

if __name__=="__main__":
    pross =  PreprocessData()
    X = pross.process()