# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 18:55:38 2021

@author: serverbob
"""
# import docker

import subprocess
import queue
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from check_woof import predict
#Some Utils
import os
import random
from threading import Thread

import playsound
playback= True # playback dog barking sounds
plot_show = True
sleep_time = 5 #seconds

if plot_show ==  True:
    from matplotlib.animation import FuncAnimation

audio_dir = './audio_files/'



def stop_docker():
    subprocess.Popen('docker stop tensor')
    
    
def play_woof():
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
    

    cmd = 'docker run --name tensor --rm -p 8501:8501 --mount type=bind,source=D:\python2\woof_friend\Dogtor-AI\models\woof_detector,target=/models/woof_detector -e MODEL_NAME=woof_detector  tensorflow/serving'    
    subprocess.Popen(cmd)
    # os.system(cmd)

def float2pcm(sig, dtype='int16'):
    """Convert floating point signal with a range from -1 to 1 to PCM.
    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.
    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them.  For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
    Parameters
    ----------
    sig : array_like
        Input array, must have floating point type.
    dtype : data type, optional
        Desired (integer) data type.
    Returns
    -------
    numpy.ndarray
        Integer data, scaled and clipped to the range of the given
        *dtype*.
    See Also
    --------
    pcm2float, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


def float_to_byte(sig):
    # float32 -> int16(PCM_16) -> byte
    return  float2pcm(sig, dtype='int16').tobytes()


# Plot audio with zoomed in y axis
def plotAudio(output):
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,10))
    plt.plot(output, color='blue')
    ax.set_xlim((0, len(output)))
    ax.margins(2, -0.1)
    plt.show()

# Plot audio
def plotAudio2(output):
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,4))
    plt.plot(output, color='blue')
    ax.set_xlim((0, len(output)))
    plt.show()

def minMaxNormalize(arr):
    mn = np.min(arr)
    mx = np.max(arr)
    return (arr-mn)/(mx-mn)





def mic_index(): #get blue yetti mic index
    count = sd.query_devices()
    print('index of available devices')
    for i, item in enumerate(count):
        try:


            
            if ("yeti" in item['name'].lower() )and ("micro" in item['name'].lower()):
                print(i,":", item['name'], "Default SR: ",item['default_samplerate'])
                sr = int(item['default_samplerate'])
                return i, sr
        except:
            pass
    
try:
    start_tf_server()
except Exception as e:
    print(e)
# time.sleep(5)


dev_mic, RATE =  mic_index()

BUFFER_SECONDS = .5  #seconds
SLIDING_WINDOW =  .45 #seconds (needs to be divisible by BUFFER_SECONDS)
FORMAT =  np.float32
# if BUFFER_SECONDS%SLIDING_WINDOW != 0:
#     raise CustomException('this is my custom message')
WIDTH =  2
n_windows =  int(1/SLIDING_WINDOW)     
RATE =  22050   
CHUNKSIZE = int(RATE*.60) #.60 of a second (.1sec overlap)

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
                            # frames_per_buffer =  CHUNKSIZE
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


data_buffer = []
# audio_buffer = []
n_times =  0
save_name = 0 
CHANNELS = 1 

buff = np.array([])
indata_1 = 0
audio_buffer = 0          

q = queue.Queue()
if plot_show ==  True:
    p = queue.Queue()

loud_threshold =  loudness()
# time.sleep(1)
save_name=1

#initialize count for dog barks
count = 0
    
def callback(indata, frames, time, status,woof= 0,):
    global count 

    # print(  indata.shape, aduio_buffer.shape)
    # print(aduio_buffer, print(np.squeeze(indata)))
    # audio_buffer =indata
    # print(CHUNKSIZE)
    global buff 
    global indata_1    
    if status:
    
        print(status)
    if any(indata):
        print(frames)
        if all(buff) == None:
            buff = np.squeeze(indata)
            # q.put(np.squeeze(indata))
        
            print("init_concat")
            
        else:
            # print(len(q.queue), "quee")
            #trim oldest audio fromt the front using CHUNKSIZE and add the newest cunk
            #to the end.
            # print("go time")
            
            # print(len(q.queue), "quee2")
            # audio_buffer = np.concatenate(( q.get()[int(frames*.5):] ,np.squeeze(indata)))

            # indata_1 = indata
            # buff  = q.get()
            buff= np.concatenate((buff[-int(frames*.25):] ,np.squeeze(indata)))
            # audio_buffer = np.concatenate(( buff[int(frames*.5):] ,np.squeeze(indata)))
            # print(len(q.queue), "quee2.2")
            # q.put(np.squeeze(indata))
            # print(len(q.queue), "quee3")
            # audio_buffer =  np.concatenate((audio_buffer, indata))
            # print(len(audio_buffer))
        
            if(np.mean(np.abs(indata))<loud_threshold):
                print("inside silence reign")   
            else:
                # global data
                #detect woof from check_woof.py
                # print(audio_buffer[np.newaxis,:].shape)
                
                audio_buffer =  buff[np.newaxis,:]
                print("length audio buffer", audio_buffer.shape)
                global data
                woof, array, data = predict(audio_buffer, confidence = .90, wording = True)
                
                
                if plot_show ==  True:
                    p.put(data)
                    
                    
            if woof == 1:
                print("woof woof a dog was heard")
                count+=1
                print(count)
                if (count == 5) & (playback == True):
                    music_thread = Thread(target=play_woof)
                    music_thread.start()
                    count = 0
                    print(f'sleeping for {sleep_time} seconds')
                    sd.sleep(int(sleep_time*1000))
                
                    

                    
                    
                    
                # print("woofanator is active")
                #activate woofanator
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
    # print(plotdata[0], type(plotdata[0]))
    # return im

        
   
    
try:
    if plot_show == True:
        plotdata = []
        fig = plt.figure()
        #initialize plot
        plotdata.append(np.random.rand(40,87))
        im =plt.imshow(plotdata[0], animated = True)
        fig.tight_layout(pad=0)
    
    stream =  sd.InputStream(device = dev_mic,
                        channels =  1,
                        samplerate = RATE,
                        callback=callback,
                        blocksize=int(RATE*2),
                        #blocksize=int(RATE * BUFFER_SECONDS),
                        )
    
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

    
# while(True):
#     #read chunk and load it into numpy array
#     data = stream.read(CHUNKSIZE)
#     current_window = np.frombuffer(data, dtype = np.float32) #from buffer reads bits 
#     #reduce noise real-time
#     current_window = nr.reduce_noise(y=current_window, sr=RATE, y_noise=noise_sample)
    
#     # n_times +=1
#     # if(n_times==50):
#     #     print(n_times)

    


    


# stream.stop_stream()
# stream.close()
# p.terminate()