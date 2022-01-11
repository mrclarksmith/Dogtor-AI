# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 18:55:38 2021
@author: Alexsey Gromov
"""
import os
import random
import subprocess
import queue
from threading import Thread

import numpy as np
import sounddevice as sd
import playsound


import tensorflow as tf
# from tensorflow_addons.optimizers import RectifiedAdam
# tf.keras.optimizers.RectifiedAdam = RectifiedAdam
def run_tensor():
    # model_path = r'D:\python2\woof_friend\models\woof_detector\1641850981'
    # model = tf.keras.models.load_model(model_path)
    tflite_model= r'./models/woof_friend_final.tflite'
    interpreter = tf.lite.Interpreter(tflite_model)
    interpreter.allocate_tensors()
        
    return interpreter


# from check_woof import import_model#import predict from same folder
from check_woof import predict
def stop_docker():
    subprocess.Popen('docker stop tensor')
    
    
def play_woof():
    #TODO change to sd.play() remove thread?
    audio_file = random.choice([x for x in os.listdir(audio_dir) if x.endswith(".mp3")] )
    print(audio_file)
    playsound.playsound(audio_dir+audio_file) 

#start tensorflow server
def start_tf_server():
    # client = docker.from_env()
    # docker.types.Mount()
    # doc_create = '-t --rm -p 8501:8501 --mount type=bind,source=D:\python2\woof_friend\Dogtor_AI\Doctor-AI\models\woof_detector,target=/models/woof_detector -e MODEL_NAME=woof_detector  tensorflow/serving'
    # client.containers.run('tensorflow/serving', entrypoint='python',
    #                                                   command='/tmp/{}/__main__.py'.format(DOCKER_BASE_FOLDER),
    #                                                   volumes=['{}:/tmp'.format(TEMP)],
    #                                                   detach=True, auto_remove=True,
    #                                                   user=uid, name=name,
    #                                                   ports={'8080/tcp': docker_config.PYWREN_SERVER_PORT})'tensorflow/serving', command= doc_create, detach=True)

    #raspbery could need to run -v vs -mount
    cmd = 'sudo docker run --name tensor --rm -p 8501:8501 -v type=bind,source=.\models\woof_detector,target=/models/woof_detector -e MODEL_NAME=woof_detector  tensorflow/serving'    
    subprocess.Popen(cmd)

def mic_index(): #get blue yetti mic index
    devices = sd.query_devices()
    print('index of available devices')
    for i, item in enumerate(devices):
        try:
            if ("yeti" in item['name'].lower() )and ("micro" in item['name'].lower()):
                print(i,":", item['name'], "Default SR: ",item['default_samplerate'])
                sr = int(item['default_samplerate'])
                return i, sr
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


#thie function needs alot of work but x2 loudness works fine for now
def loudness():
    stream = sd.InputStream(channels=1, 
                            device =  dev_mic,    
                            )
    stream.start()
    n_sample =  stream.read(int(RATE*1))[0] # reads 4 seconds of scilence
    stream.stop()
    noise_sample = n_sample
    print("Noise Sample distribution variance")
    # plotAudio2(noise_sample)
    variance(noise_sample)
    if variance(noise_sample) > .000001:
        loud_threshold =  np.mean(np.abs(noise_sample))
        noise_sample =noise_sample[np.abs(noise_sample)< loud_threshold*1]
        loud_threshold =  np.max(np.abs(noise_sample))*1
        
    else: 
        loud_threshold =  np.max(np.abs(noise_sample))*1
    print(variance(noise_sample))
    # plotAudio2(noise_sample)
    print("Loud threshold", loud_threshold)
    return loud_threshold


def update_plot(frame):
    global plotdata
    while True:
        try:
            data = p.get_nowait()
            plotdata.append(data)
            plotdata.pop(0)
        except queue.Empty:
            break
    im.set_array(plotdata[0])
    
    
    
###############################################################################
#main callback funtion for the stream : This is done in new thread per sounddevice
# NOTE: that  woof = 0 needed to set woof prediction to false 
def callback(indata, frames, time, status, woof= 0):
    global woof_count 
    global buff   
    global audio_buffer
    # global model
    if status:
        print(status)
        
    if any(indata):
        print(frames)
        if all(buff) == None:
            buff = np.squeeze(indata)
            print("init_concat")
        else:
            buff= np.concatenate((buff[-int(RATE*BUFFER_ADD):] ,np.squeeze(indata)))
            if(np.mean(np.abs(indata))<loud_threshold):
                print("inside silence reign")   
            else:
                audio_buffer =  buff[np.newaxis,:]
                print("length audio buffer", audio_buffer.shape)
                global data
                woof, array, data = predict(audio_buffer, interpreter, confidence=.93, wording = True)

                if plot_show ==  True:
                    p.put(data)                   
            if woof == 1:
                print("woof woof a dog was heard")
                woof_count+=1
                print(woof_count)
                if (woof_count == 5) & (playback == True):
                    music_thread = Thread(target=play_woof)
                    music_thread.start()
                    woof_count = 0 # reset count 
                    print(f'sleeping for {sleep_time} seconds')
                    sd.sleep(int(sleep_time*1000)) # put the sound stream to sleep
                
                #save file to desctop to analize
                # wf = wave.open('./save_audio/Aduio_clip_'+str(save_name)+".wav", 'wb')
                # t = np.linspace(0., 1., samplerate)
                # amplitude = np.iinfo(np.int16).max
                # data = amplitude * np.sin(2. * np.pi * fs * t)
                # write('./save_audio/Aduio_clip_'+str(save_name)+".wav", RATE, audio_buffer)
                # print("File/Audio saved")
                # save_name +=1
                
    else:
        print('no input')    


#####################################################################################
#Setting Initiation 
playback= True # playback dog barking sounds
DOCKER = False # docker tensorflow server does not work on arm65
plot_show = False #shows plot for every sound that activates prediction function aka a loud sound
sleep_time = 5 #seconds after the computer barks back, we sleep
BUFFER_SECONDS = 1 #Each buffer frame is analized by the tensorflow engine for dog prediction. this frame is counted in seconds + extra trim on the dge
BUFFER_ADD =.15 #Seconds to add to the buffer from previous buffer for prediction
CHANNELS = 1 #Number of audio channels (left/Right/Mono)
audio_dir = './audio_files/' #directory where the barking sounds are
CONFIDENCE = .93 #Confidence of the prediciton model for identifying if the sound contains dog bark
#Variable initiation #do not change
save_name = 0  #used for saving waves files # Not sued currently
buff = np.array([])  #Saves as global data buffer for predicting. If the bark happends at the end or beggining we ened to createa a window overlap
audio_buffer = 0    #creates an array from buffer #TODO can be combined with buff variable        
woof_count = 0 #initialize count for dog barks

#start docker server with tf    otherwise uses tensorflow light

if DOCKER == True:
    try:
        start_tf_server()
    except Exception as e:
        print(e)


# dev_mic, RATE =  mic_index()
#Detect Loudness minimum level. When loudness exceeds threshold detection for dog barks is triggered


#Set Recording Device
devices = sd.query_devices()
print(devices)
dev_mic = int(input("Enter mic number to use: "))

 #Rate of the microphone is overwritten later
RATE =  22050   # samnples per second : Setting custom rate to 22050 instead of 44100 to save on computational time
loud_threshold =  loudness()
# model = import_model()
#Loop Start #################################################################################################
try:
    if plot_show == True:
        import matplotlib.pyplot as plt #cannot be ran on raspberry pi for now
        from matplotlib.animation import FuncAnimation
        p = queue.Queue()
        plotdata = []
        fig = plt.figure()
        #initialize plot
        plotdata.append(np.random.rand(40,87)) # set initial value for the plot to get a frame size etc. This locks the plot in place for future
        im =plt.imshow(plotdata[0], animated = True)
        fig.tight_layout(pad=0)
    print("loading model")
    interpreter = run_tensor()
    
    
    stream =  sd.InputStream(device = int(dev_mic),
                        channels =  1,
                        samplerate = RATE,
                        callback=callback,
                        blocksize=int(RATE*BUFFER_SECONDS),
                        )
    
    #Sets plot to update automatically on interval of 2?? what ever that is blit False is the only way it works with the update_plot function not having a return variable at the same time. 
    if plot_show == True:
        ani = FuncAnimation(fig, update_plot, interval=2 , blit=False)
        
    with stream:
        while True:
            if plot_show == True:
                plt.show()
            
            response = input()
            if response in ('', 'q', 'Q'):
                stop_docker()
                break
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
    subprocess.Popen('docker stop tensor')  
    print("end",  e)
    
    
    










#utilities Not used
# def float2pcm(sig, dtype='int16'):
#     """Convert floating point signal with a range from -1 to 1 to PCM.
#     Any signal values outside the interval [-1.0, 1.0) are clipped.
#     No dithering is used.
#     Note that there are different possibilities for scaling floating
#     point numbers to PCM numbers, this function implements just one of
#     them.  For an overview of alternatives see
#     http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
#     Parameters
#     ----------
#     sig : array_like
#         Input array, must have floating point type.
#     dtype : data type, optional
#         Desired (integer) data type.
#     Returns
#     -------
#     numpy.ndarray
#         Integer data, scaled and clipped to the range of the given
#         *dtype*.
#     See Also
#     --------
#     pcm2float, dtype
#     """
#     sig = np.asarray(sig)
#     if sig.dtype.kind != 'f':
#         raise TypeError("'sig' must be a float array")
#     dtype = np.dtype(dtype)
#     if dtype.kind not in 'iu':
#         raise TypeError("'dtype' must be an integer type")

#     i = np.iinfo(dtype)
#     abs_max = 2 ** (i.bits - 1)
#     offset = i.min + abs_max
#     return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


# def float_to_byte(sig):
#     # float32 -> int16(PCM_16) -> byte
#     return  float2pcm(sig, dtype='int16').tobytes()


# Plot audio with zoomed in y axis
# def plotAudio(output):
#     fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,10))
#     plt.plot(output, color='blue')
#     ax.set_xlim((0, len(output)))
#     ax.margins(2, -0.1)
#     plt.show()

# # Plot audio
# def plotAudio2(output):
#     fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,4))
#     plt.plot(output, color='blue')
#     ax.set_xlim((0, len(output)))
#     plt.show()

# def minMaxNormalize(arr):
#     mn = np.min(arr)
#     mx = np.max(arr)
#     return (arr-mn)/(mx-mn)


