# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 11:04:06 2021

@author: serverbob
"""
import os
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy as sp

FRAME_SIZE = 1024
HOP_LENGTH =  512

file1="file1.mp3"
file2="19-adam_cooper_ft_janey_brown-acid_rain_(vlad_rusu_remix).mp3"
file3= ""


# ipd.Audio(file1)

lfile1, sr = librosa.load(file2)
lfile1= lfile1[:5000]
# lfile2, _ = librosa.load(file2)
# sample_duration = 1/sr
# print(f'Duration of 1 sample is: {sample_duration:.8f} seconds')

# duration = lfile1.size * sample_duration
# print(f'Duration of is: {duration:.4f} seconds')
# frames =np.array( range(0,len(lfile1),HOP_LENGTH))
# t =  frames/sr

ft = sp.fft.fft(lfile1)

plt.subplot()
plt.plot(ft.real,ft.imag)



magnitude = np.absolute(ft)
frequency = np.linspace(0, sr, len(magnitude))




plt.figure(figsize = (18,8))
plt.plot(range(len(magnitude)), magnitude)



# # t =  librosa.frames_to_time(frames, hop_length = HOP_LENGTH)

# def amplitude_envelope(signal, frame_size, hop_length):
#     amplitude_envelope=[]

#     #calculate AE for each frame
#     for i in range(0, len(signal), hop_length):
#         current_frame_amplitude_envelope =  max(signal[i:i+frame_size])
#         amplitude_envelope.append(current_frame_amplitude_envelope)
#     # print(amplitude_envelope)   
#     return np.array(amplitude_envelope)


# def fancy_amplitude_envelope(signal, frame_size, hop_length):
#     return np.array([max(signal[i:i+frame_size]) for i in range (0 , len(signal), hop_length)])



# def rms(signal, frame_size, hop_length):
#     rms_envelope=[]


#     for i in range(0, len(signal), hop_length):
#         current_rms=  np.sqrt(np.sum(signal[i:i+frame_size]**2)/frame_size)
#         rms_envelope.append(current_rms)
#     # print(amplitude_envelope)   
#     return np.array(rms_envelope)
    
    
    
# # tic = time.perf_counter()
# # are_f1 = amplitude_envelope(lfile1, FRAME_SIZE, HOP_LENGTH)
# # print(f"time regular {time.perf_counter() - tic:0.4f}")

# # tic = time.perf_counter()
# # are_ff1 = fancy_amplitude_envelope(lfile1, FRAME_SIZE, HOP_LENGTH)
# # print(f"time fancy {time.perf_counter() - tic:0.4f}")

# # plt.figure(figsize=(15,5))

# # plt.subplot(3,1,1)
# # librosa.display.waveplot(lfile1, alpha= .5)
# # plt.plot(t, are_f1, color= 'r')
# # plt.title("file1(nunez follow me)")
# # plt.ylim(-1,1)


# # plt.subplot(3,1,2)
# # librosa.display.waveplot(lfile2, alpha= .5 )
# # plt.title("file1(nunez follow me)")
# # plt.ylim(-1,1)





# rms_dfile1 = rms(lfile1,FRAME_SIZE,HOP_LENGTH)    
    
    
# rms_file1 =  librosa.feature.rms(lfile1, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]

# plt.figure(figsize=(18,8))

# plt.subplot(3,1,1)
# librosa.display.waveplot(lfile1, alpha= .5)
# plt.plot(t, rms_dfile1, color= 'g')
# plt.title("file1(nunez follow me)")
# plt.ylim(-1,1)


# # plt.subplot(3,1,2)
# # librosa.display.waveplot(lfile2, alpha= .5 )
# # plt.title("file1(nunez follow me)")
# # plt.ylim(-1,1)





    
    