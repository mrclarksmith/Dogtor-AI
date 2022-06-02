import os
import librosa
import numpy as np
from autoencoder import VAE
import random
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf
'''
creates mel spectrogram, needs work on generating mels for training, right now has both old and new mothod of picle

'''

FRAME_LENGTH = 32 #NUMBER OF TIME SAMPLES
FRAME_HEIGHT = 96 #Number of mel channels

DESIRED_SAMPLES = FRAME_LENGTH*256

LEARNING_RATE = 0.0005
BATCH_SIZE = 64
EPOCHS = 200
SAMPLES = 400

FRAME_LENGTH = 32
FRAME_WIDTH= 96


def graph(f1,f2,f3):
    fig, (ax1, ax2, ax3) = plt.subplots(3,)
    
    img1 = librosa.display.specshow(f1, ax=ax1)
    img2 = librosa.display.specshow(f2, ax=ax2)
    img3 = plt.plot(range(len(f3)), f3,  'ro')
    fig.tight_layout()
    ax1.set(title='mel_spec original')
    ax2.set(title='mel_spec reconstructed')
    ax3.set(title='10 Latent Sapce')
    fig.show()



def _get_wav_files(directory):
    fu = [os.path.join(dp, f) for dp, dn, filenames in os.walk(directory) for f in filenames if
                  os.path.splitext(f)[1].lower() == '.wav']
    return fu
       

   
def _normalize(x, a=0, b= 1):
    x = ( (b-a)*( x- x.min()) / ( x.max() - x.min() ) ) +(a)
    return x

def _power_to_db(S):
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB

def _data_pad_random(item, frame_length):
    if item.shape[1] < frame_length:
        pad = int( (frame_length-item.shape[1])*random.random() )
        padded = np.pad(
                    item,(
                        (0,0),(pad,(frame_length-item.shape[1]-pad))
                        ),'constant',constant_values = (0)
                    )
        
    elif item.shape[1] > frame_length:
        before = int((item.shape[1]-frame_length)*random.random()*.3 )
        padded = item[:,before:frame_length+before]
            
    return padded    

def _data_pad(item, frame_length):
    if item.shape[1] > frame_length:
        item = item[:,:frame_length]
        return item
    elif item.shape[1] < frame_length:
        item = np.pad(
                    item,(
                        (0,0),(0,(frame_length-item.shape[1]))
                        ),'constant',constant_values = (0)
                    )
        return item
    return item

def preprocess(filename, desired_samples=DESIRED_SAMPLES, load=True):
    sr = 22050
    N_FFT = 1024
    HOP_LENGTH = N_FFT//4
    # n_mfcc = FRAME_HEIGHT              
    # window_type = 'hann' 
    # feature = 'mel'     
    frame_lenght= FRAME_LENGTH #NUMBER OF TIME SAMPLES
    frame_height = FRAME_WIDTH #NUMBER OF MELS
    FMIN = 125
    if load:        
        audio , _ = librosa.load(filename, res_type='kaiser_fast', sr=sr, mono=True)
        audio  = np.trim_zeros(audio)
        audio = audio[:desired_samples] 
        # audio = np.pad(audio, (0, desired_samples-audio.size) , mode='constant')
    else:
        audio = filename
    mel = librosa.feature.melspectrogram(audio,sr, 
                                         n_fft=N_FFT,
                                         hop_length=HOP_LENGTH, 
                                         n_mels=frame_height, 
                                         htk=False, 
                                         fmin=FMIN
                                        )

    # Taking the magnitude of the STFT output  
    mel = _power_to_db(mel)
    mel = _normalize(mel)
    mel = _data_pad(mel, frame_lenght)

    return mel


def process_data(mylist):
    result1 = [] # Mel frequency
    num_items =  len(mylist)
    print("generating mels", num_items)
    for i, item in enumerate(mylist):
        if i % 200 == 0:
            print(i)
        r1 = preprocess(item)
        result1.append(r1)
         
    result1 = np.dstack(result1) #convert list to np array
    result1 = np.rollaxis(result1,-1)        #bring last axis to front
    result1 = np.expand_dims(result1,-1) # add channel of 1 for conv2d to work
    # result1 = result1[:,(100-FRAME_WIDTH):,::]    
    return result1



# def preprocess(pickle_path, pickle_path2, pickle_path3, FRAME_LENGTH):
#     with open(pickle_path, "rb") as f:
#         mfcc_list = pd.DataFrame(pickle.load(f))
#     with open(pickle_path2, "rb") as f:
#         mfcc_list2 = pd.DataFrame(pickle.load(f))
#     with open(pickle_path3, "rb") as f:
#         mfcc_list3 = pd.DataFrame(pickle.load(f))           
    
    
#     mfcc_combined = pd.con.t([mfcc_list.T, mfcc_list2.T, mfcc_list3.T], axis = 0)
#     mfcc_combined[1] = mfcc_combined[1].astype(int)
#     mfcc_combined = mfcc_combined.loc[mfcc_combined[1]==3]
#     X = list(mfcc_combined[0])
        
#     for i, item in enumerate(X):
#         item = power_to_db(item)
#         X[i] = normalize(item)

    
#     X_train = data_set(X, train=True)
#     X = np.array(X_train)    
#     return X 



# def data_set(X,train = False):
#     X_1 = []
#     #Pad the data
#     for row, item in enumerate(X):
#         if item.shape[1] <FRAME_LENGTH:
            
#             pad = int( (FRAME_LENGTH-item.shape[1])*random.random() )
                       
#             X_1.append(
#                     np.pad(
#                         item,(
#                             (0,0),(pad,(FRAME_LENGTH-item.shape[1]-pad))
#                             ),'constant',constant_values = (0)
#                         )
#                     )
            

#         elif item.shape[1] >FRAME_LENGTH:
#             before = int((item.shape[1]-FRAME_LENGTH)*random.random()*.3 )
#             X_1.append( item[:,before:FRAME_LENGTH+before] )

            
#             X_1.append( item[:,-(FRAME_LENGTH+before+1):-before-1] )
#         else:
#             X_1.append(item)
            
    
#     X_1 = np.dstack( X_1) #convert list to np array
#     X_1 = np.rollaxis( X_1,-1)        #bring last axis to front
#     X_1 = np.expand_dims( X_1,-1) # add channel of 1 for conv2d to work
#     X_1 = X_1[:,(100-FRAME_WIDTH):,::]
#     return  X_1

def VAE_load(latent_space=10):
    autoencoder = VAE(
        input_shape=(FRAME_WIDTH, FRAME_LENGTH, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(1, 2, 2, 2, (2, 1)),
        latent_space_dim=latent_space
    )
    return autoencoder

def train(x_train, learning_rate, batch_size, epochs):
    print("py1")
    autoencoder.summary()
    print("py2")
    autoencoder.compile(learning_rate)
    print("py3")
    autoencoder.train(x_train, batch_size, epochs)
    print("py4")
    return autoencoder

def train_s(x_train, learning_rate, batch_size, epochs):

    print("py4")
    return autoencoder

def data_out(samples=SAMPLES):
    dog_dir = "D:/python2/woof_friend/Dogtor-AI/ML/data/ML_SET/dog"
    files = _get_wav_files(dog_dir)
    x_train = process_data(files[:samples])
    return x_train     


if __name__ == "__main__":
    dog_dir = "D:/python2/woof_friend/Dogtor-AI/ML/data/ML_SET/dog"
    files = _get_wav_files(dog_dir)
    x_train = process_data(files[:SAMPLES])
    autoencoder = VAE_load(latent_space=10)
    
    
    # autoencoder.load_model_tf(save_folder="model_no_eager_test_full3") 
    # autoencoder._build()
    
    
    # autoencoder.save(save_folder="model_test2")
    # autoencoder.load(save_folder="model_test2")
    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
        
    # autoencoder.model.compute_output_shape(input_shape=[128,96,32,1])

    # autoencoder.save_model("model_no_eager_test_full3", f="tf")   
    # autoencoder.train(x_train, BATCH_SIZE, EPOCHS)
    
    # autoencoder = VAE_load(latent_space=10)
    # plot_real_vs_fake_stft(x_train[0:2], x_train[2:4])   #needs to be a list


    # autoencoder.save_model("model_no_eager_test_full")
    # plot_model(autoencoder.encoder, to_file='model.png') 
    
    
    

    reconstructed, latenent_rep = autoencoder.reconstruct(x_train[4][np.newaxis, ...])
    graph(np.squeeze(x_train[4]),np.squeeze(reconstructed), np.squeeze(latenent_rep))
    # from autoencoder import L__sample_point_from_normal_distribution
    # import tensorflow as tf
    # autoencoder = tf.keras.models.load_model("./model_no_eager_test_full/full_model.h5", 
    #                            custom_objects={'L__sample_point_from_normal_distribution': L__sample_point_from_normal_distribution})
