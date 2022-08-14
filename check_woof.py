# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 12:02:53 2021
@author: Alexsey Gromov
"""
import librosa
import numpy as np
import time  # used for checking timing only

SR = 22050
N_FFT = 512
N_MELS = 96  # Height of the PICTURE (data points per time frame)
WINDOW_TYPE = 'hann'
FEATURE = 'mel'
HOP_LENGTH = N_FFT//2
FRAME_LENGTH = 97  # Maximum frames that TF model can take can take (Frames=time Frames)
FMIN = 200
PAD = 20  # Pad Length in the beggining if ther is more than 20 frames to pasd


def normalize(x):
    b = 1
    a = 0
    y = ((b-a)*(x - x.min()) / (x.max() - x.min())) + (a)
    # x /= np.max(np.abs(x),axis=0)
    return y


def power_to_db(S):
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB


def run_lite_model(X, interpreter):
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], X)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


def extract_features_mel(file_name, wav=False, sr=22050):
    try:
        if wav == True:
            file_name, _ = librosa.load(file_name, res_type='kaiser_fast', sr=sr, mono=True)
    except:
        return None
    try:
        mfccs = librosa.feature.melspectrogram(file_name, sr,
                                               n_fft=N_FFT,
                                               hop_length=HOP_LENGTH,
                                               n_mels=N_MELS,
                                               htk=False,
                                               fmin=FMIN,)
        return mfccs
    except Exception as e:
        print("Error encountered getting mel spectrogram: ", e)
        return None

# Pad the data because frame length that was coded into the machine learning model cannot change


def pad_data(data):
    # print(data.shape)
    # pad data before and after for better algorithm detection
    if data.shape[1] < (FRAME_LENGTH-PAD):
        data = np.pad(
            data, ((0, 0), (PAD, FRAME_LENGTH-data.shape[1]-PAD)), 'constant', constant_values=(0))
    if data.shape[1] < FRAME_LENGTH:
        data = np.pad(
            data, ((0, 0), (0, FRAME_LENGTH-data.shape[1])), 'constant', constant_values=(0))
    elif data.shape[1] > FRAME_LENGTH:
        data = data[:, :FRAME_LENGTH]
    return data


def predict(audio_buffer, interpreter, confidence=.93, wav=False, sr=22050, additional_data=True):
    # audio_buffer set up to analize multiple frames at the same time by passing a bach of mels into tensorflow

    mfcc_m = []
    for audio in audio_buffer:
        data = extract_features_mel(audio, wav=wav, sr=sr)
        data = power_to_db(data)
        data = normalize(data)
        data_padded = pad_data(data)

        mfcc_m.append(data_padded)

    data_padded_m = np.array(mfcc_m)
    X = data_padded_m[..., np.newaxis]
    X = np.array(X, np.float32)
    try:
        predictions = run_lite_model(X, interpreter)
    except Exception as e:
        print(e)
        print('error prediciton is set to zero value 0')
        # capture any big exception during rollout #do not activate until final version
        predictions = [[0]]
    prediction = round(predictions[0][0], 4)

    # Returns, True false prediction score, the MEL spectrogram used for prediction
    if additional_data == True:
        if (prediction > confidence):  # TF predicts in batches must specify we want first input of the multi input batch
            return True, prediction, data  # Returns Unpadded Data
        else:
            return False, prediction, data
    else:
        return prediction


# def run_tensor(X):
#     model = tf.keras.models.load_model(model_path)
#     return model.predict(X)

# def extract_features(file_name, wav = False, sr=44100, n_mfcc=40):
#     try:
#         if wav  == True:
#             file_name, sample_rate = librosa.load(file_name, res_type='kaiser_fast', sr = 44100, mono =True)


#         mfccs = librosa.feature.mfcc(file_name,sr, n_mfcc=n_mfcc,
#                                             n_fft=N_FFT,
#                                             hop_length=HOP_LENGTH,
#                                             n_mels=N_MELS,
#                                             htk=True
#                                            )
#         # mfccsscaled = np.mean(mfccs.T,axis=0)
#         return mfccs
#     except Exception as e:
#         print("Error encountered while parsing file: ", file_name, e)
#         return None

# def make_prediction(instances):
#    url = 'http://localhost:8501/v1/models/woof_detector:predict'
#    data = json.dumps({"signature_name": "serving_default", "instances": instances.tolist()})
#    headers = {"content-type": "application/json"}
#    json_response = requests.post(url, data=data, headers=headers)
#    predictions = json.loads(json_response.text)['predictions']
#    return predictions


# def save for later prdicting various dog barks
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
