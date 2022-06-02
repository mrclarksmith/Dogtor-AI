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
import librosa
# importing custom python module 
from check_woof import predict
from scipy.signal import hilbert

# class audioPros:

#     def __init__(self, audio, sr=22050):
#         self.audio = audio
#         self.sr = sr

#     def time_streach(self, rate):
#         '''
#         rate = Stretch factor
#         '''
#         self.audio = librosa.effects.time_stretch(self.audio, rate=rate)

#     def pitch(self,n_steps):
#         self.audio = librosa.effects.pitch_shift(self.audio, sr=self.sr, n_steps=n_steps)
    
#     def trim_scilence(self):
#         ''' trim scilence based on threshold'''

#     def _rms_mean_energy(self):
#         #[sum(x)/^2]/n
#         #rms = np.mean(np.power(self.audio,2))
#         rmse = librosa.feature.rmse(self.audio, frame_length=2048, hop_length=512, center=True, pad_mode='reflect')
#         return rmse
#     def _zero_crossing_rate(self):
#         zero_crossing = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512, center=True)
#         return zero_crossing

#     def _short_time_energy(self):
        
#         if (n > 0) and (n< N-1): 
#             h_n =  0.54-0.46 * np.cos(2*np.pi * n / (N-1) )
#             np.sum(y)

#     def _signal_covering(self):
#         h = hilbert(self.audio)
#         # |phi| = [ g(t)^2 + g_hat(t)^2 ]^(1/2)
#         return abs(np.sqrt(np.power(self.audio,2)+np.power(h,2)))

#     def dynamic_threshold(self):
#         E = self._rmse_mean_energy()
#         Z = self._zero_crossing_rate()
#         l_e = (max(E) - min(E)) / max(E)
#         l_z = (max(Z) - min(Z)) / max(Z)

#         E_th = (1-l_e) * max(E) + l_e * min(E)
#         Z_th = (1-l_z) * max(Z) + l_z * min(Z)



#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6068870/


# darwin = macOS, win32 = Windows
if sys.platform in "darwin win32":
    from playsound import playsound
    print("macOS or Windows detected, using playsound")
else:
    from sound_player import SoundPlayer, Sound

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
def loudness(dev_mic):
    stream = sd.InputStream(channels=1,
                            device=dev_mic,
                            )
    stream.start()
    n_sample = stream.read(int(RATE*2))[0]  # reads 4 seconds of scilence
    stream.stop()
    stream.close()
    
    loud_treshold = loudness_thresh_calc(n_sample)
    
    return loud_treshold
    

def loudness_thresh_calc(n_sample):
    sample_range = []
    for i in range(int(len(n_sample)/100)):
        sample_range.append( np.abs(n_sample[i*100:(i+1)*100]).max() )
    
    sample_range.sort()
    sample_range = sample_range[:int(len(n_sample)/100*.7)]
    loud_threshold = round(max(sample_range)*3, 4)
    
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
    print("Loud threshold rounded", loud_threshold)
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
    global save_buff
    global put_in_queue
    global p
    global flag_save

    if status:
        print(status, "status")
    if any(indata):
        if all(save_buff) is None:
            save_buff =  np.squeeze(indata)
            print("init_concat", frames)
        else:
            save_buff = np.concatenate((save_buff[-int(RATE*.5):], np.squeeze(indata)))
            buff = save_buff[-int(RATE*(BUFFER_SECONDS+BUFFER_ADD)):]
            if put_in_queue == True:
                p.put(np.squeeze(indata))
                
            indata_loudness = round(max(np.abs(indata))[0],4)
            if indata_loudness < loud_threshold:
                print("inside silence reign:", "Listening to buffer",frames,"samples", "Loudness:", indata_loudness)
            else:
                woof, prediction, data = predict(buff[np.newaxis, :], interpreter, confidence=.93, additional_data=True)
                print("Predictions: score:", prediction, "Loudness:", indata_loudness, "/", loud_threshold)
                if (prediction > .70) and ( SAVEAUDIO is True) and (flag_save == True):
                    put_in_queue = True 
                    flag_save = False
                    save_thread = Thread(target=save_audio, args=(save_buff, f"_P{round(prediction,4)}L{max(np.abs(indata))}"))
                    save_thread.start()   
                if woof == 1:
                    th_w = Thread(target=thread_woof)
                    th_w.start()
    else:
        print('no input')
#####################################################################################


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

def main():

    stream = sd.InputStream(device = dev_mic,
                        channels = 1,
                        samplerate = RATE,
                        callback=callback,
                        blocksize=int(RATE*BUFFER_SECONDS),
                        )
    
        #Sets plot to update automatically on interval of 2?? what ever that is blit False is the only way it works with the update_plot function not having a return variable at the same time. 
        # if PLOT_SHOW == True:
        #     ani = FuncAnimation(fig, update_plot, interval=2 , blit=False)
    try:    
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
    

if __name__ == "__main__":
    
    # Setting Initiation
    VERSION = '0.0.3'
    SAVEAUDIO = True # Save each trigger of audio to a file
    PLAYBACK = False  # PLAYBACK dog barking sounds
    DOCKER = False  # docker tensorflow server does not work on arm65
    PLOT_SHOW = True  # shows plot for every sound that activates prediction function aka a loud sound
    SLEEP_TIME = 5  # seconds after the computer barks back, we sleep
    BUFFER_SECONDS = 1.25  # Each buffer frame is analized by the tensorflow engine for dog prediction. this frame is counted in seconds + extra trim on the edge. Max+buffer Add = 2 seconds
    BUFFER_ADD = .15  # Seconds to add to the buffer from previous buffer for prediction, cannot exceed 2 seconds combined with BUFFER_SECONDS
    CHANNELS = 1  # Number of audio channels (left/Right/Mono) #not configurable
    AUDIO_DIR = './audio_files/'  # Directory where the barking sounds are
    CONFIDENCE = .88  # Confidence of the prediciton model for identifying if the sound contains dog bark
    RATE = 22050  # Samples per second : Setting custom rate to 22050 instead of 44100 to save on computational time #Rate of the microphone is overwritten later. Big dudu will happend if changed and you will not even know
    REC_AFTER =2 # NUmber x Buffer_seconds to record after the event has occured
    # Variable initiation #do not change
    save_name = 0  # Used for saving waves files # Not sued currently
    save_buff =  np.array([])
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
    parser.add_argument('--LOUDNESS', dest='LOUDNESS', type=float)

    args = parser.parse_args()

    print(f"Dogtor AI version {VERSION} initializing...")
    print("output values may be rounded")
    
    #Set Recording Device
    devices = sd.query_devices()
    print(devices)
    if args.MIC is None:
        dev_mic = int(input("Enter mic number to use: "))
    else:
        dev_mic = mic_index(args.MIC)


    if args.LOUDNESS is None:
        loud_threshold = loudness(dev_mic)
        if loud_threshold == 0:
            print("Mic is off or not working, try different driver or mic")
        print("loudness set to ambient noise")
    else:
        loud_threshold = args.LOUDNESS
        print("loudness set to:", args.LOUDNESS)

    print("loading model")
    interpreter = run_tensor()
    print("loading sound input")
    main()
