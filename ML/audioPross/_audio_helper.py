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
from shutil import copy
from tkinter import filedialog, Tk
from pydub import AudioSegment
from pydub.silence import split_on_silence
from scipy.fft import fft, fftfreq
from noise_remove import removeNoise
import pyloudnorm as pyln

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




def select_folder():
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    try:
        folder_selected = filedialog.askdirectory()
    except:
        print("file error")
    return folder_selected

def select_file():
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    try:
        file_selected =  filedialog.askopenfilename()
        print(file_selected)
    except:
        print("file error")
    return file_selected






def extract_file_name():
    try:
        folder_selected = select_folder()
        print(folder_selected)
    except:
        print("read folder error")
        exit()

    try:
        save_directory = select_folder()
        print(save_directory)
    except:
        print("save folder error")
        exit()
        
    for root, _ , files in os.walk(folder_selected, topdown=False):
        print(root)
        for name in files:
            try:
                category = name.split("-")[1]
                if category != "3":
                    copy(os.path.join(root,name), os.path.join(save_directory,name))
            except Exception as e:
                print(e)
    print("done")


#stop  prevents from generating all the file at 
def extract(stop=None):
    mfcc_list =  []
    category_list = []
    file_list = []
    count = 1 

    DIR_USER = select_folder()
    
    for root, _ , files in os.walk(DIR_USER, topdown = False):
        for name in files:
            count +=1
            if count == stop:
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
                                            hop_length=N_FFT//2 , 
                                            n_mels=N_MELS, 
                                            htk=False, 
                                            # fmin=FMIN
                                           )    
    
    mfccs3 = librosa.feature.melspectrogram(y,sr=SR,
                                            n_fft=512,
                                            hop_length=N_FFT//1 , 
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


# Define a function to normalize a chunk to a target amplitude.
def match_target_amplitude(aChunk, target_dBFS):
    ''' Normalize given audio chunk '''
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)


def chunk_audio(silence_length = 200, silence_thresh=-30, scilence=6, num_chunk=False):
    dir_dog = select_folder()
    if not num_chunk:
        save_dir = select_folder()
    num_c = 0
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
                    min_silence_len = silence_length,
                    # Consider a chunk silent if it's quieter than -16 dBFS.
                    # (You may want to adjust this parameter.)
                    silence_thresh = silence_thresh
                )
                print(name)
                if num_chunk:
                    num_c += len(chunks)
                    print(num_c)
                # Process each chunk with your parameters
                else:
                    for i, chunk in enumerate(chunks):
                        # Create a silence chunk that's 0.5 seconds (or 500 ms) long for padding.
                        silence_chunk = AudioSegment.silent(duration=scilence)
        
                        # Add the padding chunk to beginning and end of the entire chunk.
                        audio_chunk = silence_chunk + chunk + silence_chunk
        
                        # Normalize the entire chunk.
                        # normalized_chunk = match_target_amplitude(audio_chunk, -20.0)
        
                        # Export the audio chunk with new bitrate.
                        # print(f"Exporting chunk{i}")
                        name = name.split(".")[0]
                        audio_chunk.export(
                            f"{save_dir}/chunk{name}_{i}.wav", format = "wav")
    

def del_unwanted_audio():
    folder= r'D:\python2\woof_friend\Dogtor-AI\ML\data\NewlinRecording2203'
    destination = r'D:\python2\woof_friend\Dogtor-AI\ML\data\NewlinRecording2203\chunk'
    silence_chunk_time= 50 #ms
    SR = 22050 
    for root, _ , files in os.walk(folder, topdown = False):
        for name in files:
            
            rms  = rms_mean_energy(folder, name, SR, silence_chunk_time)
            print(rms, name)
            if rms < 1.6e-04:
                shutil.move(folder+'/'+name, destination+'/'+name)

    

def stack():
    folder = select_folder()
    for root, _ , files in os.walk(folder, topdown=True):

        for file in files:
            file_name = os.path.join(root, file)
            # sr = 48000
            
            y, sr = load_audio_y(file_name)
            
            y = normalize(y)
            # print(len(y),sr)
            rms = round(rms_mean_energy(y, sr, 6), 5)
            
            dur = round(librosa.get_duration(y),3)
            z = _zero_crossing_rate(y)
            z_mean = round(np.mean(z),3)
            z_max = round(np.max(z),3)
            cm = round(variance(y),6)
            y_off = round(np.mean(y),4)
            
            yf = round(fft_100_262_db(y), 2)
            try:
                loud = round(loudness(y, sr),1)
            except Exception as e:
                print(file, '  error: ', e)
                loud="nan"
            os.rename(file_name, os.path.join(root,file.split("_")[0]+"_"+str(dur)[2:]+"_l_"+str(loud)[1:]+"_rms_"+str(rms)[2:]+"_cm_"+str(cm)[2:]+"_m_"+str(z_mean)[2:]+"_max_"+str(z_max)[2:]+"_off_"+str(y_off)[2:]+"_db_"+str(yf)[1:]+".wav")) 
    

def stack1(name, plt_fig=0):
    root = r"D:\python2\woof_friend\DataSets\UrbanSound8K\UrbanSound8K\dog_chunk/"
    
    
    y = load_audio_y(os.path.join(root,name))
    y = normalize(y)
    plot_audio(y)  
    y = fft(y)
    # y = np.abs(sftf(y))
    plot_fft(y, n_fft=2048, sr=44100, plt_fig=plt_fig)
    
    return y
  
    
def loudness(y, sr=48000):
    meter = pyln.Meter(sr, block_size=.1) # create BS.1770 meter
    loudness = meter.integrated_loudness(y[..., np.newaxis]) # measure loudness
    return loudness



def fft_100_262_db(y, cutoff_hz=6100):
    
    # name=r'chunk155311-3-0-0_603_rms_01769_cm_01202_m_052_max_073_off_.0.wav'
    # root = r"D:\python2\woof_friend\DataSets\UrbanSound8K\UrbanSound8K\dog_chunk/"
    # y = load_audio_y(os.path.join(root,name))  
    yf = fft(y)
    yf = np.abs(yf)
    f = fftfreq(len(y), 1/44100)[:len(y)//2]
    f_bin  = min(f, key=lambda x:abs(x-cutoff_hz))
    f_bin =  np.where(f ==f_bin)[0][0]
    yf = librosa.power_to_db(yf, ref=1.0, amin=1e-10, top_db=100)
    yf = yf[:f_bin]
    yf = yf - max(yf)
    # plt.plot(f, yf[:len(y)//2])
    # plt.xscale("log")
    yf =np.mean(yf[30:80])

    
    
    return yf
    

def sftf(y, n_fft=2048):
    y = librosa.stft(y, n_fft=n_fft)
    return y
            
def variance(data):
    data = np.abs(data)
    # Number of observations
    n = len(data)
    # Mean of the data
    mean = sum(data) / n
    # Square deviations
    deviations = [(x - mean) ** 2 for x in data]
    # Variance
    variance = sum(deviations) / n
    return variance

def save_wav(path,y,sr):
    librosa.output.write_wav(path, y, sr, norm=False)

def plot_audio(y):
    plt.figure(120)
    plt.plot(y)


def plot_fft(y,n_fft, sr, plt_fig=0):
    plt.figure(plt_fig)
    # y_avg =  np.mean(y, axis=1)
    plt.plot(y)
    x_ticks_positions = [n for n in range(0, n_fft // 2, n_fft // 16)]
    x_ticks_labels = [str(sr / 2048 * n) + 'Hz' for n in x_ticks_positions]
    plt.xticks(x_ticks_positions, x_ticks_labels)
    plt.xscale('log')
    plt.xlabel('Frequency')
    plt.ylabel('dB')
    plt.show()

def rms_mean_energy(y, SR=22050, silence_chunk_time=0):
    # y , _ = librosa.load(name, res_type='kaiser_fast', sr = SR,  mono = True)
    # #[sum(x)/^2]/n
    rms = np.mean(np.power(y[silence_chunk_time*SR//1000:-silence_chunk_time*SR//1000],2) )
    return rms

def _zero_crossing_rate(y):
    zero_crossing = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512, center=True)
    return zero_crossing


def normalize(y):
    y = librosa.util.normalize(y)
    return y

def rmss(audio, step_size):
    rmss = []
    for position in range(0,len(audio),step_size):
        rms = np.mean(np.power(audio[position:position+step_size],2) )
        rmss.append(rms)
    return rmss
    
def load_audio_y(file, SR = None, mono=True):
    y, sr = librosa.load(file, res_type='kaiser_fast', sr = SR,  mono = mono)
    return y, sr
    

def dx_rms(xs):
    dxs = np.subtract(np.array(xs)[:-1], np.array(xs)[1:])
    return list(dxs)

def chunk_max(audio, step_size):
    chunk_maxes = []
    rmss = []
    for position in range(0,len(audio),step_size):
        rms = np.mean(np.power(audio[position:position+step_size],2) )
        chunk_maxes.append(rms)
    return chunk_maxes
def plot_rms(folder, file_name):
    
    y = load_audio_y(os.path.join(folder,file_name))
    x = range(0,len(y),220)
    rms_data = rmss(y, 220)
    dxs =  dx_rms(rms_data)
    ddxs =  dx_rms(dxs)
    fig, ax = plt.subplots(nrows=3,figsize=(20, 4), sharex=True)
    data_plot, = ax[0].plot(x,rms_data, label="Std. power")
    dx, =ax[0].plot(x[:-1],dxs, label="change in RSM")
    dx, =ax[1].plot(y, label="change in RSM")
    dx, =ax[2].plot(x[:-1], dxs, label="change in RSM")
    dx, =ax[2].plot(x[:-2], ddxs, label="change in RSM")
    
    ax[0].set_title("Threshold for mask")
    plt.show()
    
    plt.figure()
    plt.tight_layout()
    plt.hist(np.array(rms_data), bins=100, fmt="0")
    label = "{:.2f}".format(y)
    
    plt.legend()
    plt.show()


def clean_audio():
    audio = os.path.join("D:\python2\woof_friend\Dogtor-AI\ML\data\Recording_dog.wav")
    SR = 48000
    noise_sample_dir = os.path.join('D:/python2/woof_friend\Dogtor-AI\ML/data/Noise_dog.wav')
    y, sr = librosa.load(audio, sr=SR)
    noise_sample, _ = librosa.load(noise_sample_dir, sr=SR)
    
    output = removeNoise(y, noise_sample,
                          n_std_thresh=2,
                          prop_decrease=0.95,
                          visual=True,
                          )
    return output
     
    
# import sounddevice as sd

# sd.play(output)
# sd.play(noise_sample)
# sd.play(y)


import soundfile as sf
def save_signals(signals, save_dir, sample_rate=22050):
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, str(i) + ".wav")
        sf.write(save_path, signal, sample_rate)
        
        
# save_dir = os.path.join(r"D:\python2\woof_friend\Dogtor-AI\ML\data")        
# audio_no_noise = clean_audio()

# save_signals([audio_no_noise,],save_dir,SR)
# y = stack1(name = r'chunk29936-3-1-0_315_rms_01961_cm_010925_m_082_max_115_off_e-04.wav', plt_fig=0) 
# y= stack1("chunk28284-3-0-1_614_rms_01476_cm_009274_m_151_max_225_off_.0.wav", plt_fig=1)

