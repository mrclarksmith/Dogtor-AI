# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 19:30:52 2021

@author: Server
"""

import librosa 
import os

import soundfile as sf 
rootdir = "D:/python2/woof_friend/UrbanSound8K/UrbanSound8K/audio/"
OUTPUT_DIR =  'D:/python2/woof_friend/UrbanDog/'
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file[-4:] == ".wav":
            # filename = os.path.splitext(file)[0]
            y, sr = librosa.load(subdir + "/" + file, sr=22050)  # Downsample 44.1kHz to 22.05kH
            y = librosa.to_mono(y)
            sf.write(OUTPUT_DIR + file, y , samplerate = sr, format = "WAV")





DOG_DIR =  'D:/python2/woof_friend/UrbanDog_Only/'

from shutil import copy
for filename in os.listdir(OUTPUT_DIR):
    if filename.split("-")[-3] == "3":
        copy(os.path.join(OUTPUT_DIR,filename), DOG_DIR)
