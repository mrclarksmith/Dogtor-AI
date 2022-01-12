import sounddevice as sd
import playsound


def mic_index(): #get blue yetti mic index
    devices = sd.query_devices()
    print('index of available devices')
    for i, item in enumerate(devices):
        try:
            print(i,":", item['name'], "Default SR: ",item['default_samplerate'])
        except:
            pass
    
mic_index()


stream = sd.InputStream(channels=1, 
                            device =  1,    
                            )
stream.start()
n_sample =  stream.read(int(44100*1))[0] # reads 4 seconds of scilence
stream.close()
mic_index()

stream = sd.InputStream(channels=1, 
                            device =  1,    
                            )
stream.start()
n_sample =  stream.read(int(44100*1))[0] # reads 4 seconds of scilence
stream.stop()
mic_index()
