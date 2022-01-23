# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 12:02:53 2021

@author: serverbob
"""

import requests
import json
import librosa
import numpy as np
#from sklearn.metrics import accuracy_score
# sound = './one_woof.wav'

# from tensorflow.keras.models import Model
# import tflite_runtime.interpreter as tflite
# import tensorflow as tf
# from tensorflow_addons.optimizers import RectifiedAdam
# tf.keras.optimizers.RectifiedAdam = RectifiedAdam

SR = 22050
N_FFT = 1024        
HOP_SIZE = 1024       
N_MELS = 100            
WIN_SIZE = 1024      
WINDOW_TYPE = 'hann' 
FEATURE = 'mel'      
FMIN = 20 #minimum frequency
FMAX = 4000
HOP_LENGTH = N_FFT//2

#model_path = r'D:\python2\woof_friend\models\woof_detector\1641850981'
#tflite_model= r'D:\python2\woof_friend\woof_friend3.tflite'
# model_path ='D:/python2/woof_friend/c_f_model.h5',

# import tensorflow as tf
# model2 = tf.saved_model.load(model_path, compile=False)
# model = tf.keras.models.load_model(model_path, compile=False)
# interpreter = tflite.Interpreter(model_path=model_path)
def normalize(x, a=0, b= 1):
    y = ( (b-a)*( x- x.min()) / ( x.max() - x.min() ) ) +(a)
    return y

def power_to_db(S):
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB

def run_lite_model(X, interpreter):

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], X)
    # interpreter.set_tensor(input_details[0]['index'], np.array(X_test[:100], dtype = np.float32))
    # interpreter.resize_tensor_input(input_details[0]['index'], (100,100,176,1))
    # interpreter.resize_tensor_input(output_details[0]['index'], (100,1))
    # interpreter.allocate_tensors()
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # print(output_data)
    
    return output_data
    
    
def run_tensor(X):
    model = tf.keras.models.load_model(model_path)
    return model.predict(X)

def extract_features(file_name, wav = False, sr=44100, n_mfcc=40):
    try:
        if wav  == True:
            file_name, sample_rate = librosa.load(file_name, res_type='kaiser_fast', sr = 44100, mono =True)
            

        mfccs = librosa.feature.mfcc(file_name,sr, n_mfcc=n_mfcc,
                                            n_fft=N_FFT,
                                            hop_length=HOP_SIZE, 
                                            n_mels=N_MELS, 
                                            htk=True, 
                                            fmin=FMIN
                                           )
        # mfccsscaled = np.mean(mfccs.T,axis=0)
        return mfccs
    except Exception as e:
        print("Error encountered while parsing file: ", file_name, e)
        return None 
     

def extract_features_mel(file_name, wav = False, sr=22050):
    try:
        if wav  == True:  
            file_name, _ = librosa.load(file_name, res_type='kaiser_best', sr = sr, mono =True)
    except:
        return None
    try:
        mfccs = librosa.feature.melspectrogram(file_name,sr, 
                                                n_fft=N_FFT,
                                                hop_length=HOP_LENGTH, 
                                                n_mels=N_MELS, 
                                                htk=False, 
                                                # fmin=FMIN
                                               )
        return mfccs 
    except Exception as e:
        print("Error encountered while parsing file: ",e)
        return None 
    
#Frame length that was coded into the machine learning model cannot change     
frame_length = 97
def pad_data(data):

    print(data.shape)
    #pad data before and after for better algorithm detection
    if data.shape[1] <(frame_length-20):
        data = np.pad(data,((0,0),(20,frame_length-data.shape[1]-20)),'constant',constant_values = (0))
        
    if data.shape[1] <frame_length:
        data = np.pad(data,((0,0),(0,frame_length-data.shape[1])),'constant',constant_values = (0))
    elif data.shape[1] >frame_length:
        data = data[:][:frame_length]
    return data



def make_prediction(instances):
   url = 'http://localhost:8501/v1/models/woof_detector:predict'
   data = json.dumps({"signature_name": "serving_default", "instances": instances.tolist()})
   headers = {"content-type": "application/json"}
   json_response = requests.post(url, data=data, headers=headers)
   predictions = json.loads(json_response.text)['predictions']
   return predictions



def predict(audio_buffer, interpreter, confidence=.93, wav = False, sr=22050, wording = False):
    #audio_buffer set up to analize multiple frames at the same time by passing a bach of mels into tensorflow
    mfcc_m = []

    for audio in audio_buffer:
        data = extract_features_mel(audio, wav=wav, sr=sr)
        data = power_to_db(data)
        data = normalize(data)
        data_padded = pad_data(data)
        mfcc_m.append(data_padded)
        print(data_padded.shape)
    data_padded_m = np.array(mfcc_m)
       
    X = data_padded_m[..., np.newaxis]
    # print(confidence)
    
    X = np.array(X, np.float32)
    
    try:
        predictions = run_lite_model(X, interpreter)
    except Exception as e:
        print(e)
        print('error prediciton is set to zero value 0')
        prediction = [[0]] #capture any big exception during rollout #do not activate until final version
        
    print(predictions)
    #predictions  = make_prediction(X)
   
 

    prediction = predictions[0][0]

    if wording == True:
        if (prediction> confidence ):#TF predicts in batches must specify we want first input of the multi input batch 
            # print("woof woof")
            print("Predictions: score: ", prediction)
        
            return 1, predictions, data
        else:
            print("Predictions: score: ", prediction)
            # print("we predict the number:" ,np.argmax(predictions) )
            return 0, predictions, data
        
    ans = predictions[0][0]    
    return ans


        
#def save for later prdicting various dog barks
    # ans = np.argwhere(predictions == np.amax(predictions, 1, keepdims = True))
    # if wording == True:
    #     if ( np.argmax(predictions[0]) == 3 )and (predictions[0][3]> confidence ):#TF predicts in batches must specify we want first input of the multi input batch 
    #         # print("woof woof")
    #         print("Predictions: score: ", predictions[0][3])
        
    #         return 1, predictions, data
    #     else:
    #         print("Predictions: score: ", predictions[0][3])
    #         # print("we predict the number:" ,np.argmax(predictions) )
    #         return 0, predictions, data
    # return ans
        

# def show_mel(q):
#     # fig, ax = plt.subplots(nrows=10,ncols=1, figsize=(20,4))

#     # ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=1000)
#     print("plot")
#     global plotdata
#     try:
#         if len(plotdata<5):
#             plotdata.append(q)
#         else:
#             plotdata.pop(0)
#             plotdata.append(q)
#     except:
#         plotdata.append(q)
        
#     for i, item in enumerate(plotdata):
#         ax[i] = plt.imshow(item)
    
#     fig.canvas.draw()

    
# def update_plot():
#     global plotdata
#     # while True:
#     try: 
#         data =  q.get_nowwait()
#     except queue.Empty:
#         break
#     while len(plotdata<5)
#         poltdata.append(data)
#     else:
#         plotdata.pop(0)
#         plotdata.append(data)
    

#     # plt.plot(output, color='blue')
#     # ax.set_xlim((0, len(output)))
#     plt.show()

#     FuncAnimation
    


        