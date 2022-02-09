# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 21:48:57 2022

@author: server
"""

import unittest
import listen_woof
import check_woof
import numpy as np
import librosa 
import librosa.display
import matplotlib.pyplot as plt



def normalize(x, a=0, b= 1):
    y = ( (b-a)*( x- x.min()) / ( x.max() - x.min() ) ) +(a)
    return y
def plot_mel(S):
    sr = 22050
    # N_FFT = 1024 
    # WIN_LENGTH = N_FFT
    SR = 22050
    N_FFT = 512
    N_MELS = 100
    WINDOW_TYPE = 'hann'
    FEATURE = 'mel'
    # HOP_LENGTH = N_FFT//1
    HOP_LENGTH = N_FFT//1 
    mel = librosa.feature.melspectrogram(S,22050, 
                                            n_fft=N_FFT,
                                            hop_length=HOP_LENGTH, 
                                            n_mels=N_MELS, 
                                            htk=False,)
    mel = power_to_db(mel+1e-5)
    imgplot = plt.imshow(mel)
    # librosa.display.specshow(S, sr=sr,  x_axis='time', y_axis='mel');
    # plt.colorbar(format='%+1.0f dB');  
    return imgplot
    
def power_to_db(S):
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB


def test_data():
    return np.random.rand(40,87)

def test_audio():
    buff = np.random.rand(26000,).astype(dtype=np.float32)
    audio_buffer = buff[np.newaxis, :]
    return  audio_buffer

def test_audio_one():
    return  np.random.rand(26000,)




class Unit_Test(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        filename1 = "./test/chunk2021-06-30_00-13-07.wav"
        filename2 = "./test/chunk2021-07-09_04-40-23.wav"
        self.indata_dog1, _ = librosa.load(filename1, res_type = 'kaiser_best', sr = 22050, mono = True)
        self.indata_dog2, _ = librosa.load(filename2, res_type = 'kaiser_best', sr = 22050, mono = True)
        self.indata_noise =  test_audio_one()
        self.interpreter = listen_woof.run_tensor()
        

    def setUp(self):
        pass
    
    def test_predict(self):
        woof, prediction, data = check_woof.predict(self.indata_dog1[np.newaxis, :], self.interpreter, confidence=.93, additional_data=True)
        print(prediction)
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        imgplot = plt.imshow(data)
        ax = fig.add_subplot(1, 2, 2)
        imgplot = plot_mel(self.indata_dog1)
        self.assertGreater(prediction, .70)
        
    def test_predict1(self):
        woof, prediction, data = check_woof.predict(self.indata_dog2[np.newaxis, :], self.interpreter, confidence=.93, additional_data=True)
        print(prediction)
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        imgplot = plt.imshow(data)
        ax = fig.add_subplot(1, 2, 2)
        imgplot = plot_mel(self.indata_dog2)
        self.assertGreater(prediction, .70)
        
    def test_predict2(self):
        woof, prediction, data = check_woof.predict(self.indata_noise[np.newaxis, :], self.interpreter, confidence=.93, additional_data=True)
        print(prediction)
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        imgplot = plt.imshow(data)
        ax = fig.add_subplot(1, 2, 2)
        imgplot = plot_mel(self.indata_noise)
        
        
        
        
        
        
        
    
if __name__ == "__main__":
    unittest.main()
