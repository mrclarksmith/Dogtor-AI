#!/usr/bin/env python
"""
Created on Wed Jan 26 17:30:47 2022

@author: server
"""
# import random
# from itertools import count
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# plt.style.use('fivethrtyeight')


# def animate(i):
#     x_vals.append(next(index))
#     y_vals.append(random.randint(0,5))
#     plt.cla()
#     plt.plot(x_vals, y_vals)
#     plt.legend(loc='upper left')
#     plt.tight_layout()


# ani =  FuncAnimation(plt.gcf(), animate, 1000)



# plt.tight_layout()
# plt.show()
# x_vals = []
# y_vals = []
# index = count()


from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

import sys
from scipy.fftpack import fft
import sounddevice as sd



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


class AudioStream(object):
    def __init__(self):

        # pyqtgraph stuff
        pg.setConfigOptions(antialias=True)
        self.traces = dict()
        self.app = QtGui.QApplication(sys.argv)
        self.win = pg.GraphicsWindow(title='Spectrum Analyzer')
        self.win.setWindowTitle('Spectrum Analyzer')
        self.win.setGeometry(5, 115, 1910, 1070)

        wf_xlabels = [(0, '0'), (2048, '2048'), (4096, '4096')]
        wf_xaxis = pg.AxisItem(orientation='bottom')
        wf_xaxis.setTicks([wf_xlabels])

        wf_ylabels = [(0, '0'), (127, '128'), (255, '255')]
        wf_yaxis = pg.AxisItem(orientation='left')
        wf_yaxis.setTicks([wf_ylabels])

        sp_xlabels = [
            (np.log10(10), '10'), (np.log10(100), '100'),
            (np.log10(1000), '1000'), (np.log10(22050), '22050')
        ]
        sp_xaxis = pg.AxisItem(orientation='bottom')
        sp_xaxis.setTicks([sp_xlabels])

        self.waveform = self.win.addPlot(
            title='WAVEFORM', row=1, col=1, axisItems={'bottom': wf_xaxis, 'left': wf_yaxis},
        )
        self.spectrum = self.win.addPlot(
            title='SPECTRUM', row=2, col=1, axisItems={'bottom': sp_xaxis},
        )
        
        # pyaudio stuff
        # self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024*4
        self.NUMCHUNK = 20
        self.buffer =np.zeros(self.CHUNK * self.NUMCHUNK)
        self.stream = sd.InputStream(samplerate=self.RATE, blocksize = self.CHUNK, device =  2, channels = 1, dtype = None )
        self.stream.start()
        # waveform and spectrum x points
        self.x = np.arange(0, self.CHUNK*self.NUMCHUNK, 1)
        self.f = np.linspace(0, self.RATE, self.CHUNK)
        self.count = 0

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()
        print("you just closed the pyqt window!!! you are awesome!!!")



    def set_plotdata(self, name, data_x, data_y, ):
        if name in self.traces:
            if name== 'waveform':
                self.buffer = self.buffer[self.CHUNK:]
                self.buffer =  np.append(self.buffer,data_y)
                self.traces[name].setData(data_x, self.buffer)
            else:
                self.traces[name].setData(data_x, data_y)
        else:
            if name == 'waveform':
                self.traces[name] = self.waveform.plot(pen='c', width=3)
                self.waveform.setYRange(-1, 1, padding=0)
                self.waveform.setXRange(0, self.CHUNK*self.NUMCHUNK, padding=0.005)
            if name == 'spectrum':
                self.traces[name] = self.spectrum.plot(pen='m', width=3)
                self.spectrum.setLogMode(x=True, y=True)
                self.spectrum.setYRange(-8, 0, padding=0)
                self.spectrum.setXRange(
                    np.log10(20), np.log10(self.RATE/2), padding=0.005)
        if self.count == 20:
            try: 
                self.waveform.removeItem(self.item)
            except:
                pass
            temp = self.loudness()
            
            self.item = pg.InfiniteLine(temp, angle = 0)
            self.waveform.addItem(self.item)
            # temp =  QLine()
            # self.waveform.plot(x = [0,10000], y = [temp, temp])
            
            self.count = 0 
        self.count +=1 
        
        
        
    def update(self):
        wf_data = self.stream.read(self.CHUNK)[0].reshape(self.CHUNK)
        self.set_plotdata(name='waveform', data_x=self.x, data_y=wf_data,)

        sp_data = fft(np.array(wf_data, dtype='float32'))
        sp_data = np.abs(sp_data[0:int(self.CHUNK)]
                         )/ (self.CHUNK)
        self.set_plotdata(name='spectrum', data_x=self.f, data_y=sp_data)

    def animation(self):
        
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(20)
        self.start()

    def loudness(self):
        print("Noise Sample distribution variance")
        # plotAudio2(noise_sample)

        if variance(self.buffer) > .000001:
            loud_threshold = np.mean(np.abs(self.buffer))
            noise_sample =self.buffer[np.abs(self.buffer)< loud_threshold*1]
            loud_threshold = np.max(np.abs(noise_sample))*1
            
        else: 
            loud_threshold = np.max(np.abs(self.buffer))*1
        print(variance(self.buffer))
        # plotAudio2(noise_sample)
        print("Loud threshold", loud_threshold)
        return loud_threshold
    
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




if __name__ == '__main__':

    audio_app = AudioStream()
    audio_app.animation()

    
    