#!~/prog/DogPI/bin/python

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 18:55:38 2021
@author: Alexsey Gromov
"""
import argparse
import os
import queue
import random
import sys
import numpy as np
import sounddevice as sd
import wave
from threading import Thread
from pathlib import Path
import tensorflow as tf
import time
# from check_woof import import_model#import predict from same folder
from check_woof import predict

# darwin = macOS, win32 = Windows
if sys.platform in "darwin win32":
    from playsound import playsound
    print("macOS or Windows detected, using playsound")
else:
    from sound_player import SoundPlayer, Sound

# Setting Initiation
VERSION = '0.0.2'
SAVEAUDIO = True # Save each trigger of audio to a file
PLAYBACK = False  # PLAYBACK dog barking sounds
DOCKER = False  # docker tensorflow server does not work on arm65
PLOT_SHOW = True  # shows plot for every sound that activates prediction function aka a loud sound
SLEEP_TIME = 5  # seconds after the computer barks back, we sleep
BUFFER_SECONDS = 1  # Each buffer frame is analized by the tensorflow engine for dog prediction. this frame is counted in seconds + extra trim on the edge. Max+buffer Add = 2 seconds
BUFFER_ADD = .15  # Seconds to add to the buffer from previous buffer for prediction, cannot exceed 2 seconds combined with BUFFER_SECONDS
CHANNELS = 1  # Number of audio channels (left/Right/Mono) #not configurable
AUDIO_DIR = './audio_files/'  # Directory where the barking sounds are
CONFIDENCE = .88  # Confidence of the prediciton model for identifying if the sound contains dog bark
RATE = 22050  # Samples per second : Setting custom rate to 22050 instead of 44100 to save on computational time #Rate of the microphone is overwritten later. Big dudu will happend if changed and you will not even know
REC_AFTER =2 # NUmber x Buffer_seconds to record after the event has occured
# Variable initiation #do not change
save_name = 0  # Used for saving waves files # Not sued currently
buff = np.array([])  # Saves as global data buffer for predicting. If the bark happends at the end or beggining we ened to createa a window overlap
save_buff =  np.array([])
audio_buffer = 0  # Creates an array from buffer #TODO can be combined with buff variable
woof_count = 0  # Initialize count for dog barks
p = queue.Queue(1)
put_in_queue = False # Indicates if que recording is to start
flag_save = True # Indicates if a save process is running not to duplicate the sounds (queue management)
parser = argparse.ArgumentParser(description='options')
parser.add_argument('--SAVEAUDIO', dest='SAVEAUDIO', type=bool)
parser.add_argument('--PLAYBACK', dest='PLAYBACK', type=bool)
parser.add_argument('--DOCKER', dest='DOCKER', type=bool)
parser.add_argument('--PLOT_SHOW', dest='PLOT_SHOW', type=bool)
parser.add_argument('--SLEEP_TIME', dest='SLEEP_TIME', type=float, help='sleep after a bark')
parser.add_argument('--BUFFER_SECONDS', dest='BUFFER_SECONDS', type=float)
parser.add_argument('--BUFFER_ADD', dest='BUFFER_ADD', type=float)
parser.add_argument('--CONFIDENCE', dest='CONFIDENCE', type=float)
parser.add_argument('--MIC', dest='MIC', type=str)

args = parser.parse_args()

print(f"Dogtor AI version {VERSION} initializing...")


def run_tensor():
    tflite_model= r'./models/woof_friend_final.tflite'
    interpreter = tf.lite.Interpreter(tflite_model)
    interpreter.allocate_tensors()
    return interpreter


# 2 version one for raspberry one for windows
if sys.platform in "darwin win32":
    def play_woof():
        # TODO change to sd.play() remove thread?
        audio_file = random.choice([x for x in os.listdir(AUDIO_DIR) if x.endswith(".mp3")])
        print(audio_file)
        playsound(AUDIO_DIR+audio_file)
else:
    def play_woof():
        audio_file = random.choice([x for x in os.listdir(AUDIO_DIR) if x.endswith(".mp3")])
        print(AUDIO_DIR+audio_file)
        player = SoundPlayer()
        player.enqueue(Sound(AUDIO_DIR+audio_file), 1)
        player.play()


def mic_index(dev_name):  # get blue yetti mic index
    devices = sd.query_devices()
    print('index of available devices')
    for i, item in enumerate(devices):
        try:
            if (dev_name in item['name'].lower()) and ( item['max_output_channels']>0):
                print(i, ":", item['name'], "Default SR: ", item['default_samplerate'])
                # sr = int(item['default_samplerate'])
                return i
        except:
            pass


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


# This function needs alot of work but x2 loudness works fine for now
def loudness():
    stream = sd.InputStream(channels=1,
                            device=dev_mic,
                            )
    stream.start()
    n_sample = stream.read(int(RATE*2))[0]  # reads 4 seconds of scilence
    stream.stop()
    stream.close()
    
    sample_range = []
    for i in range(int(len(n_sample)/100)):
        sample_range.append( np.abs(n_sample[i*100:(i+1)*100]).max() )
    
    sample_range.sort()
    sample_range[:int(len(n_sample)/100*.4)]
    loud_threshold = max(sample_range)*1.23
    
    print("Noise Sample distribution variance")
    # plotAudio2(noise_sample)
    # variance(noise_sample)
    # if variance(noise_sample) > .000001:
    #     loud_threshold = np.mean(np.abs(noise_sample))
    #     noise_sample = noise_sample[np.abs(noise_sample) < loud_threshold*1]
    #     loud_threshold = np.max(np.abs(noise_sample))*1

    # else:
    #     loud_threshold = np.max(np.abs(noise_sample))*1
    # print(variance(noise_sample))
    # plotAudio2(noise_sample)
    print("Loud threshold", loud_threshold)
    return loud_threshold

# def update_plot(frame):
#     global plotdata2
#     while True:
#         try:
#             data = p.get_nowait()
#             plotdata.append(data)
#             plotdata.pop(0)
#         except queue.Empty:
#             break
#     im.set_array(plotdata[0])


def save_audio(data, name):
    global flag_save
    global put_in_queue
    for i in range(REC_AFTER):
        p_get =  p.get()
        data = np.concatenate((data,p_get))
    now = str(round(time.time()))
    path_s = Path('./audiosave/')
    path_s.mkdir(parents=True, exist_ok=True)
    path = name+".wav"
    max_16bit = 2**15
    # print("dir made")
    data = data * max_16bit
    data = data.astype(np.int16)
    with wave.open('./audiosave/'+now+path,mode='w') as wb:
        wb.setnchannels(1)
        wb.setsampwidth(2)
        wb.setframerate(RATE)
        wb.writeframes(data)  #Convert to byte string
    print("saved")
    flag_save = True
    put_in_queue = False

        
def thread_woof():
    global woof_count
    print("woof woof a dog was heard")
    woof_count += 1
                    
    if (woof_count == 1) & (PLAYBACK == True):
        play_woof()
        # music_thread = Thread(target=play_woof)
        # music_thread.start()
        woof_count = 0  # reset count
        print(f'sleeping for {SLEEP_TIME} seconds')
        # sd.sleep(int(SLEEP_TIME*1000))  # put the sound stream to sleep   ] 

###############################################################################
# main callback funtion for the stream : This is done in new thread per sounddevice
# NOTE: that woof = 0 needed to set woof prediction to false
def callback(indata, frames, _ , status, woof= 0):
    global woof_count
    global buff
    global audio_buffer
    global PLOT_SHOW
    global save_buff
    global put_in_queue
    global p
    global flag_save
    if status:
        print(status)
    if any(indata):
        if all(buff) is None:
            buff = np.squeeze(indata)
            save_buff =  np.squeeze(indata)
            print("init_concat")
        else:
            buff = np.concatenate((buff[-int(RATE*BUFFER_ADD):], np.squeeze(indata)))
            save_buff = np.concatenate((save_buff[-int(RATE*3):], np.squeeze(indata)))
            if put_in_queue == True:
                p.put(np.squeeze(indata))
                
                
            if(np.mean(np.abs(indata)) < loud_threshold):
                pass
                print("inside silence reign:", "Listening to buffer",frames," samples")
            else:
                audio_buffer = buff[np.newaxis, :]
                woof, prediction, data = predict(audio_buffer, interpreter, confidence=.93, additional_data=True)
                prediction= .99
                if (prediction > .70) and ( SAVEAUDIO is True) and (flag_save == True):
                    put_in_queue = True
                    flag_save = False
                    save_thread = Thread(target=save_audio, args=(save_buff, f"_S{prediction}"))
                    save_thread.start()   
                # if woof == 1:
                #     print("woof = 1")
                #     th_w = Thread(target=thread_woof)
                #     th_w.start()

                

    else:
        print('no input')
#####################################################################################

#Set Recording Device
devices = sd.query_devices()
print(devices)
if args.MIC is None:
    dev_mic = int(input("Enter mic number to use: "))
else:
    dev_mic = mic_index(args.MIC)

loud_threshold =  loudness()



#Loop Start #################################################################################################
# if PLOT_SHOW == True :
#     import matplotlib.pyplot as plt #cannot be ran on raspberry pi for now
#     from matplotlib.animation import FuncAnimation
#     
#     plotdata = []
#     fig = plt.figure()
#     #initialize plot
#     plotdata.append(np.random.rand(40,87)) # set initial value for the plot to get a frame size etc. This locks the plot in place for future
#     im =plt.imshow(plotdata[0], animated = True)
#     fig.tight_layout(pad=0)

try:
    print("loading model")
    interpreter = run_tensor()
    print("loading sound input")

    stream = sd.InputStream(device = dev_mic,
                        channels = 1,
                        samplerate = RATE,
                        callback=callback,
                        blocksize=int(RATE*BUFFER_SECONDS),
                        )
    
    #Sets plot to update automatically on interval of 2?? what ever that is blit False is the only way it works with the update_plot function not having a return variable at the same time. 
    # if PLOT_SHOW == True:
    #     ani = FuncAnimation(fig, update_plot, interval=2 , blit=False)

    with stream:
        while True:
            if PLOT_SHOW == True:
                pass
            # response = input()
            # if response in ('', 'q', 'Q'):
            #     break
            # for ch in response:
            #     if ch == '+':
            #         args.gain *= 2
            #     elif ch == '-':
            #         args.gain /= 2
            #     else:
            #         print('\x1b[31;40m', usage_line.center(args.columns, '#'),
            #               '\x1b[0m', sep='')
            #         break

except Exception as e:
    #subprocess.Popen('docker stop tensor')
    print("end", e)

