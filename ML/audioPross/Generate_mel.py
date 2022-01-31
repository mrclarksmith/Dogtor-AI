# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 22:11:20 2021

@author: Server
"""
import numpy as np
import pandas as pd
import librosa
import os
import pickle 
import matplotlib.pyplot as plt
import librosa.display
import shutil 
SR = 22050
N_FFT = 512 
# WIN_LENGTH = N_FFT
HOP_LENGTH = N_FFT//1      
# HOP_SIZE = 1028       
N_MFCC = 100              
WINDOW_TYPE = 'hann' 
FEATURE = 'mel'     
N_MELS=100
# FMIN = 20 #minimum frequency
# FMAX = 4000



def extract_features_mfcc(file_name, sr=22050):
    
    y, sample_rate = librosa.load(file_name, res_type='kaiser_fast', sr = 22050, mono =True)
    mfccs = librosa.feature.mfcc(y,sr, n_mfcc=N_MFCC,
                                            n_fft=N_FFT,
                                            hop_length=HOP_LENGTH, 
                                            n_mels=N_MELS, 
                                            htk=True, 
                                            # fmin=FMIN
                                           )
    return mfccs, y


def extract_features_mel(file_name, sr=22050):
    y, sample_rate = librosa.load(file_name, res_type='kaiser_best', sr = sr, mono =True)
    mfccs = librosa.feature.melspectrogram(y,sr, 
                                            n_fft=N_FFT,
                                            hop_length=HOP_LENGTH, 
                                            n_mels=N_MELS, 
                                            htk=False, 
                                            # fmin=FMIN
                                           )
    return mfccs, y

#stop  prevents from generating all the file at 
def extract(stop=None):
    mfcc_list =  []
    category_list = []
    file_list = []
    count = 1 
    # DIR_USER = "D:/python2/woof_friend/UrbanSound8K/UrbanSound8K/audio"
    DIR_USER = r'D:\python2\woof_friend\Dogtor-AI\ML\data\6058NewlinDogsRecordings\chunk'
    
    for root, _ , files in os.walk(DIR_USER, topdown = False):
        for name in files:
            count +=1
            if count ==  stop:
                break           
            try:
                category = name.split("-")[1]
                mfccs, _ = extract_features_mel(root+'/'+name)
                mfcc_list.append(mfccs)
                category_list.append(category)
                file_list.append(root+'/'+name)
                # print(category, name,os.path.join(root,name))
            except:
                print("error!!! in the name", name)
                pass
    
            if count%20 == 0:
                print(count, "count")
        if count ==  stop:
            break
    the_list = [mfcc_list,category_list, file_list]            
    with open("D:/python2/woof_friend/pickle_list_PetDog_Sound_event", "wb") as f:
        pickle.dump(the_list, f)
    print("saved")
    # return the_list
    
    
def extract_from_file(stop=None):
    mfcc_list =  []
    category_list = []
    file_list = []
    count = 1 
    
    # DIR_USER = r'D:\python2\woof_friend\Dogtor-AI\ML\data\PetDogSoundEvent_Barking'
    # DIR_SAVE = "D:/python2/woof_friend/pickle_list_PetDog_Sound_event"
        
    DIR_USER = r'D:\python2\woof_friend\Dogtor-AI\ML\data\6058NewlinDogsRecordings\chunk'
    DIR_SAVE = r'D:/python2/woof_friend/pickle_list_Newlin_Dog_barks'
    for root, _ , files in os.walk(DIR_USER, topdown = False):
        for name in files:
            count +=1
            if count ==  stop:
                break           
            try:
                category = 3
                mfccs, _ = extract_features_mel(root+'/'+name)
                mfcc_list.append(mfccs)
                category_list.append(category)
                file_list.append(root+'/'+name)
                # print(category, name,os.path.join(root,name))
            except:
                print("error!!! in the name", name)
                pass
    
            if count%20 == 0:
                print(count, "count")
        if count ==  stop:
            break
    the_list = [mfcc_list,category_list, file_list]            
    with open(DIR_SAVE, "wb") as f:
        pickle.dump(the_list, f)
    print("saved")
    # return the_list
    
    
      
        
    
    
    
    
    
def graph():
    file1 = "7061-6-0-0.wav"
    file2 = "7383-3-0-0.wav"
    
    file = 'D:/python2/woof_friend/UrbanSound8K/UrbanSound8K/audio/fold1/'
    y, sample_rate = librosa.load(file+file1, res_type='kaiser_fast', sr = 22050, mono =True)
    mfccs = librosa.feature.melspectrogram(y,sr=SR,
                                            n_fft=512,
                                            hop_length=HOP_LENGTH, 
                                            n_mels=N_MELS, 
                                            htk=False, 
                                            # fmin=FMIN
                                           )    
    
    mfccs3 = librosa.feature.melspectrogram(y,sr=SR,
                                            n_fft=512,
                                            hop_length=HOP_LENGTH, 
                                            n_mels=N_MELS, 
                                            htk=False, 
                                            # fmin=FMIN
                                           )    
    
    
    # mfccs =librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=1024)
    mfccs2, y2 = extract_features_mel(file+file1)
    mfccs = librosa.power_to_db(mfccs, ref=np.max)
    mfccs2 = librosa.power_to_db(mfccs2, ref=np.max)
    mfccs3 = librosa.power_to_db(mfccs3, ref=np.max)
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax1)
    img2 = librosa.display.specshow(mfccs2, x_axis='time', ax=ax2)
    img3 = librosa.display.specshow(mfccs3, x_axis='time', ax=ax3)
    # fig.colorbar(img, ax=ax2)
    ax1.set(title='mel_spec')
    fig.show()
    print(mfccs.shape, len(y))
    print(mfccs2.shape, len(y2))
    print(mfccs3.shape, len(y))
def graphZ(z):


    fig, (ax1, ax2) = plt.subplots(2)
    img = librosa.display.specshow(z, x_axis='time', ax=ax1)
    # img2 = librosa.display.specshow(mfccs2, x_axis='time', ax=ax2)
    fig.colorbar(img, ax=ax2)
    ax1.set(title='MFCC')
    fig.show()


def plot():
    file1 = "7061-6-0-0.wav"
    file2 = "7383-3-0-0.wav"    
    sgram = librosa.stft(file1)
    librosa.display.specshow(sgram)
    # use the mel-scale instead of raw frequency
    sgram_mag, _ = librosa.magphase(sgram)
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=SR)
    librosa.display.specshow(mel_scale_sgram)
    




# Import the AudioSegment class for processing audio and the 
# split_on_silence function for separating out silent chunks.
from pydub import AudioSegment
from pydub.silence import split_on_silence

# Define a function to normalize a chunk to a target amplitude.
def match_target_amplitude(aChunk, target_dBFS):
    ''' Normalize given audio chunk '''
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)


def chunk_audio():
    dir_dog = r'D:\python2\woof_friend\Dogtor-AI\ML\data\6058NewlinDogsRecordings\nn'
    for root, _ , files in os.walk(dir_dog, topdown = False):
        for name in files:
                # Load your audio.
                song = AudioSegment.from_file(root+'/'+name)
    
                # Split track where the silence is 2 seconds or more and get chunks using 
                # the imported function.
                chunks = split_on_silence (
                    # Use the loaded audio.
                    song, 
                    # Specify that a silent chunk must be at least 2 seconds or 2000 ms long.
                    min_silence_len = 240,
                    # Consider a chunk silent if it's quieter than -16 dBFS.
                    # (You may want to adjust this parameter.)
                    silence_thresh = -40
                )
                print(name)
                # Process each chunk with your parameters
                for i, chunk in enumerate(chunks):
                    # Create a silence chunk that's 0.5 seconds (or 500 ms) long for padding.
                    silence_chunk = AudioSegment.silent(duration=100)
    
                    # Add the padding chunk to beginning and end of the entire chunk.
                    audio_chunk = silence_chunk + chunk + silence_chunk
    
                    # Normalize the entire chunk.
                    # normalized_chunk = match_target_amplitude(audio_chunk, -20.0)
    
                    # Export the audio chunk with new bitrate.
                    print("Exporting chunk{0}.".format(i))
                    audio_chunk.export(
                        f"{dir_dog}/chunk/chunk{name}{i}.wav", format = "wav")
    

def del_unwanted_audio():
    folder= r'D:\python2\woof_friend\Dogtor-AI\ML\data\6058NewlinDogsRecordings\chunk'
    destination = r'D:\python2\woof_friend\Dogtor-AI\ML\data\6058NewlinDogsRecordings\noise_chunk'
    silence_chunk_time= 100 #ms
    SR = 22050 
    for root, _ , files in os.walk(folder, topdown = False):
        for name in files:
            
            rms  = rms_mean_energy(folder, name)
            print(rms, name)
            if rms < 1.6e-04:
                shutil.move(folder+'/'+name, destination+'/'+name)

                

            
            
    
def rms_mean_energy(folder, name, SR = 22050, silence_chunk_time = 100):
    y , _ = librosa.load(folder+'/'+name, res_type='kaiser_fast', sr = SR,  mono = True)
    #[sum(x)/^2]/n
    rms = np.mean(np.power(y[silence_chunk_time*SR//1000:-silence_chunk_time*SR//1000],2) )
    return rms
            

