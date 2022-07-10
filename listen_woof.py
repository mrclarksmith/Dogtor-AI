#!~/prog/DogPI/bin/python

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 18:55:38 2021
@author: Alexsey Gromov
Program to bark back at those loud dogs
"""

from flask_socketio import SocketIO, emit

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt

from PIL import Image
import argparse
import os
#import queue
import random
import sys
import numpy as np
import sounddevice as sd
import wave
from threading import Thread, Event
from pathlib import Path
import tensorflow as tf
import time
import librosa
import librosa.display
from multiprocessing import Process
from multiprocessing import Queue
# importing custom python module
from check_woof import predict
from scipy.signal import hilbert


from flask import Flask, render_template, url_for, request, redirect, make_response, Response, send_file, jsonify
import json
import random

import base64
from io import BytesIO
from flask import Flask, render_template
from PIL import Image
import io

from flask import Flask
from matplotlib.figure import Figure
os.chdir(sys.path[0])

##############################################################################
app = Flask(__name__)
socketio = SocketIO(app)
thread = Thread()
thread_stop_event = Event()


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')


@socketio.on('my event')                          # Decorator to catch an event called "my event":
def test_message(message):                        # test_message() is the event callback function.
    emit('my response', {'data': 'got it!'})      # Trigger a new event called "my response"
    # that can be caught by another callback later in the program.

# Thread Class to send data to Local Server to view spectorgrams


class SendDataThread(Thread):
    def __init__(self):
        super(SendDataThread, self).__init__()

    def run_data_to_flask(self):
        global plot_url_q
        while True:
            try:
                data = plot_url_q.get_nowait()
                arr, woof_detected = data

                # Lights up a bar underneath the mel spectrogram that detects the bark
                if woof_detected:
                    w_d = 1
                else:
                    w_d = 0

                # Send Image
                # img =  Image.fromarray(arr.astype('uint8'))

                # # create file-object in memroy
                # file_object = io.BytesIO()
                # #write PNG in file-object
                # img.save(file_object, 'PNG')

                # response = make_response(json.dumps(arr.tolist()))
                # response.content_type = 'application/json'
                # socketio.emit('newnumber', {'number': json.dumps(arr.tolist())}, namespace='/test')
                socketio.emit('newnumber', {
                                            'number': json.dumps(arr.tolist()),
                                            'woof': w_d
                                            }, namespace='/test')
                print("emit")
            except queue.Empty:
                pass

    def run(self):
        self.run_data_to_flask()


@socketio.on('connect', namespace='/test')
def test_connect():
    # need visibility of the global thread object
    global thread
    print('Client connected')

    # Start the random number generator thread only if the thread has not been started before.
    # Catches exception for PYTHON 3.9 "_" added 
    try:
        if not thread.isAlive():
            print("Starting Thread")
            thread = SendDataThread()
            thread.start()
    except:
        if not thread.is_alive():
            print("Starting Thread")
            thread = SendDataThread()
            thread.start()


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled




###########################################################################
# darwin = macOS, win32 = Windows
if sys.platform in "darwin win32":
    from playsound import playsound
    print("macOS or Windows detected, using playsound")
else:
    from sound_player import SoundPlayer, Sound  # raspberry pi comatable


def load_lite_model(model_location):
    tflite_model_dir = os.path.join("./models/", model_location)
    interpreter = tf.lite.Interpreter(tflite_model_dir)
    interpreter.allocate_tensors()
    return interpreter


def run_lite_model(X, interpreter):
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], X)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


def generate_point_audio(lite_model_vae_decoder,  lite_model_gan, encoding, save=True):
    reconstructed = run_lite_model(encoding, lite_model_vae_decoder)
    # code to load model
    mel_set = reconstructed.T
    print(mel_set.shape)
    mel_set = mel_set[np.newaxis, ...]
    # max_min_f(mel_set)

    # Generate audio_predict
    audio_generated = run_lite_model(np.squeeze(mel_set)[np.newaxis, ...], lite_model_gan)
    sd.play(np.squeeze(audio_generated), samplerate=RATE)
    # save audio
    if save:
        save_audio(audio_generated, "gen_bark")


# 2 versios; one for raspberry-pi, the other one for windows
if sys.platform in "darwin win32":
    def play_woof():
        print("playing Woof")
        # TODO change to sd.play() remove thread?
        audio_file = random.choice([x for x in os.listdir(AUDIO_DIR) if x.endswith(".mp3")])
        print(audio_file)
        playsound(AUDIO_DIR+audio_file)
else:  # raspberry pi comatable
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
            if (dev_name in item['name'].lower()) and (item['max_output_channels'] > 0):
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
                            samplerate=RATE,
                            )
    stream.start()
    n_sample = stream.read(int(RATE*4))[0]  # reads 4 seconds of scilence
    stream.stop()
    stream.close()

    loud_treshold = loudness_thresh_calc(n_sample)

    return loud_treshold


def loudness_thresh_calc(n_sample):
    sample_range = []
    for i in range(int(len(n_sample)/100)):
        sample_range.append(np.abs(n_sample[i*100:(i+1)*100]).max())

    sample_range.sort()
    sample_range = sample_range[:int(len(n_sample)/100*.7)]
    loud_threshold = round(max(sample_range)*3, 4)

    print("Loud threshold rounded", loud_threshold)
    return loud_threshold


def save_generated_bark(data, name):
    now = str(round(time.time()))
    path_s = Path('./audiosave/')
    path_s.mkdir(parents=True, exist_ok=True)
    path = name+".wav"
    max_16bit = 2**15
    # print("dir made")
    data = data * max_16bit
    data = data.astype(np.int16)
    with wave.open('./audiosave/'+now+path, mode='w') as wb:
        wb.setnchannels(1)
        wb.setsampwidth(2)
        wb.setframerate(RATE)
        wb.writeframes(data)  # Convert to byte string
    print("saved")


def save_audio(data, name):
    global flag_save
    global put_in_queue
    for i in range(REC_AFTER):
        p_get = p.get()
        data = np.concatenate((data, p_get))
    # put_in_queue = False
    save_generated_bark(data, name)

    flag_save = True

def thread_play_woof(generate):
    global woof_count
    print("dog was heard")
    woof_count += 1

    if (woof_count == WOOF_ACTIVATION_PLAYBACK_COUNT) & (PLAYBACK == True):
        if generate == True:
            # array = BARK_ENCODING
            array = np.array([np.random.normal(loc=0, scale=1, size=10)],  # use scale 2 to 10 for additonal variation
                             dtype=np.float32)  # Random Bark variation
            array = BARK_ENCODING + .5*array  # add variation
            print("playing generated audio")
            generate_point_audio(vae_lite_model, gan_lite_model, array, save=False)
        else:
            play_woof()

        woof_count = 0  # reset count after a dog bark is played

###############################################################################
# main callback funtion for the stream : This is done in new thread per sounddevice
# NOTE: that variable woof = 0 is needed to set woof prediction to false, this is so that you can disable playback of dog bark if too many barks happend 

tok_tok = 0 
def callback(indata, frames, _, status, woof=False):
    global woof_count
    global save_buff
    global put_in_queue
    global p
    global flag_save
    global plot_url_q
    global tok_tok
    if status:
        print(status, "status")
    if any(indata):
        if all(save_buff) is None:
            save_buff = np.squeeze(indata)
            print("init_concat", frames)
        else:
            # Save buffer keeps previous buffers sound to be able to join to together with present to\
            # dog bark if it happened inbeween buffer frames.
            # SAVEBUFF + indata 
            # [......|||] + [|||||||] = .5 Last Seconds + .9 Second  

            # save_buff is generic buffer that can be used to pass audio to save function to save a full clip
            save_buff = np.concatenate((save_buff[-int(RATE*.5):], np.squeeze(indata)))
            # Takes the end of the buffer for most recent audio from general buffer
            buff = save_buff[-int(RATE*(BUFFER_SECONDS+BUFFER_ADD)):]

            # //To be deleted
            # Que stops beeing filled if a bark is heard, this is disabled, que is always filled
            if put_in_queue == True:
                p.put(np.squeeze(indata))

            # only triggers Tensorflow if there is loud bark / audio
            indata_loudness = round(max(np.abs(indata))[0], 4)
            if indata_loudness < loud_threshold:
                print("inside silence reign:", "Listening to buffer",
                      frames, "samples", "Loudness:", indata_loudness)
                plot_url_q.put([np.zeros((77,96)), 0])

            else:
                # woof get sets to "True" if woof is heard
                woof, prediction, data = predict(
                    buff[np.newaxis, :], interpreter, confidence=CONFIDENCE, additional_data=True)
                print("Predictions: score:", prediction, "Loudness:",
                      indata_loudness, "/", loud_threshold)
                # put "data" from prediction in que trimmed for current frame of audio
                #print(len(indata), buff.shape, data.shape)
                # Cuts off Extra buffer to make the audio seamless

                plot_url_q.put([data.T[:77], woof])
                if (prediction > .70) and (SAVEAUDIO is True) and (flag_save == True):
                    put_in_queue = True
                    flag_save = False
                    save_thread = Thread(target=save_audio, args=(
                        save_buff, f"_P{round(prediction,4)}L{max(np.abs(indata))}"))
                    save_thread.start()
                if woof:
                    th_w = Thread(target=thread_play_woof, args=[True])  # arg:generate = True
                    th_w.start()
    else:
        print('no input')

#####################################################################################


#Loop Start #########################################################################

    # else:
    #     fig = Figure()
    #     ax = fig.subplots()

    #     ax.specgram(arr, NFFT=512, Fs=22050)

    #     buf = io.BytesIO()
    #     fig.savefig(buf, format="jpeg")
    #     encoded_img_data = base64.b64encode(buf.getbuffer())
    #     return render_template("index.html", img_data=encoded_img_data.decode('utf-8'), woof_detected=w_d)


# @app.route('/image_data')
# def image():
#     # my numpy array
#     arr = plot_url_q.get()

#     # convert numpy array to PIL Image
#     img = Image.fromarray(arr.astype('uint8'))

#     # create file-object in memory
#     file_object = io.BytesIO()

#     # write PNG in file-object
#     img.save(file_object, 'PNG')

#     # move to beginning of file so `send_file()` it will read from start
#     file_object.seek(0)

#     return send_file(file_object, mimetype='image/PNG')

#     # if queue1.empty():
#     #     pass
#     # data = plot_url_q.get()
#     # img_data = encoded_img_data = base64.b64encode(data.getvalue())
#     # img_data=base64.b64encode(data.getvalue().decode('utf8'))
#     # return render_template("index.html", img_data=img_data)


def main_stream():
    try:
        with sd.InputStream(device=dev_mic,
                            channels=1,
                            samplerate=RATE,
                            callback=callback,
                            blocksize=int(RATE*BUFFER_SECONDS)
                            ):
            print('#' * 80)
            print('press Return to quit')
            print('#' * 80)
            input()
    except KeyboardInterrupt:
        pass
        # parser.exit('')
    except Exception as e:
        pass
        # parser.exit(type(e).__name__ + ': ' + str(e))


if __name__ == "__main__":
    # Setting Initiation
    VERSION = '0.0.3'
    SAVEAUDIO = False  # Save each trigger of audio to a file
    GENERATE_WOOF = True  # Generates bark sound using tensorflow lite models
    PLAYBACK = True  # PLAYBACK dog barking sounds
    WOOF_ACTIVATION_PLAYBACK_COUNT = 1  # Number of barks the program hears before barking back
    DOCKER = False  # docker tensorflow server does not work on arm65
    PLOT_SHOW = True  # shows plot for every sound that activates prediction function aka a loud sound
    SLEEP_TIME = 0  # seconds after the computer barks back, we sleep
    # Cannot Exceed 1.1261678004 Seconds for Analysis 
    BUFFER_SECONDS = .9  # Each buffer frame is analized by the tensorflow engine for dog prediction. this frame is counted in seconds + extra trim on the edge. Max+buffer Add = 2 seconds
    BUFFER_ADD = .226  # Seconds to add to the buffer from previous buffer for prediction, cannot exceed 2 seconds combined with BUFFER_SECONDS
    CHANNELS = 1  # Number of audio channels (left/Right/Mono) #not configurable
    AUDIO_DIR = './audio_files/'  # Directory where the barking sounds are to be played back
    CONFIDENCE = .75  # Confidence of the prediciton model for identifying if the sound contains dog bark
    RATE = 22050  # Samples per second : Setting custom rate to 22050 instead of 44100 to save on computational time #Rate of the microphone is overwritten later. Big dudu will happend if changed and you will not even know
    REC_AFTER = 2  # NUmber x Buffer_seconds to record after the event has occured
    BARK_ENCODING = np.array([[-3.0, -1.6774293,  0.5526688,  7.012168, -2.2925243,
                               5.7915726,  -1.7413237,  3.5634975, -3.0460133,  -3.57509345]],
                             dtype=np.float32) # Bark encoding gotten from the real dog file as a basis for generating new barks
    WEB = True  # Sets state if to fun Flask Server
    PLOT_URL_DATA_SIZE = int(22050 * BUFFER_SECONDS / 256) # slice of MEL spectrogram to send to web server that represents current time frame 256 is hop size from check_woof.py  
    
    INTERPRETER_DIR = "woof_friend_final.tflite"
    VAE_LITE_MODEL_DIR = "vae_model_decoder_tflite.tflite"
    GAN_LITE_MODEL_DIR = "gan_model_tflite.tflite"
    # Variable initiation #DO NOT CHANGE!#
    save_name = 0  # Used for saving waves files # Not sued currently
    save_buff = np.array([])
    woof_count = 0  # Initialize count for dog barks
    # p = queue.Queue(1)
    # data_que = queue.Queue(5)
    # plot_url_q = queue.Queue(3) # que flask server to send to website

    p = Queue(1)
    data_que = Queue(5)
    plot_url_q = Queue(3) # que flask server to send to website
    put_in_queue = False  # Indicates if que recording, it gets desabled during audio save, and enabled again after
    # Indicates if a save process is running not to duplicate the sounds (queue management)
    flag_save = True
    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('--SAVEAUDIO', dest='SAVEAUDIO', type=bool)
    parser.add_argument('--PLAYBACK', dest='PLAYBACK', type=bool)
    parser.add_argument('--GENERATE_WOOF', dest='GENERATE_WOOF', type=bool,
                        help='uses tf to generate random barking sound')
    parser.add_argument('--DOCKER', dest='DOCKER', type=bool)
    parser.add_argument('--PLOT_SHOW', dest='PLOT_SHOW', type=bool)
    parser.add_argument('--SLEEP_TIME', dest='SLEEP_TIME', type=float, help='sleep after a bark')
    parser.add_argument('--BUFFER_SECONDS', dest='BUFFER_SECONDS', type=float)
    parser.add_argument('--BUFFER_ADD', dest='BUFFER_ADD', type=float)
    parser.add_argument('--CONFIDENCE', dest='CONFIDENCE', type=float)
    parser.add_argument('--MIC', dest='MIC', type=str)
    parser.add_argument('--LOUDNESS', dest='LOUDNESS', type=float)
    parser.add_argument('--WEB', dest='WEB', type=bool)

    args = parser.parse_args()

    print(f"Dogtor AI version {VERSION} initializing...")
    print("output values may be rounded")

    # Set Recording Device
    devices = sd.query_devices()
    print(devices)
    print("setting mic..")
    if args.MIC is None:
        dev_mic = input("Enter mic number to use:")
        dev_mic = int(dev_mic)
        print("Mic selected:", dev_mic)
    else:
        print("Mic was entered at run time")

    if args.LOUDNESS is None:
        loud_threshold = loudness(dev_mic)
        if loud_threshold == 0:
            print("Mic is off or not working, try different driver or mic")
        print("loudness set to ambient noise", loud_threshold)
    else:
        loud_threshold = args.LOUDNESS
        print("loudness set to:", args.LOUDNESS)
    print("Loading 3 Tensorflow Light Models...")
    interpreter = load_lite_model(INTERPRETER_DIR)
    vae_lite_model = load_lite_model(VAE_LITE_MODEL_DIR)
    gan_lite_model = load_lite_model(GAN_LITE_MODEL_DIR)

    # app.run()  # runs Flask
    # main_thread = Thread(target=main_stream)
    # main_thread.start()
    p_main = Process(target=main_stream)
    p_main.start()
    # socketio.start_background_task(main_stream)
    print("Starting Socket")
    if WEB:
       socketio.run(app, host='0.0.0.0', port=5000)


# needs to be exported to alternate module
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


# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6068870/
