from cProfile import label
from cmath import cos
import dataclasses
import math
from turtle import xcor
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import scipy.signal as signal
from scipy.signal import butter, lfilter, freqz
import os
#import acoustics 

def raspi_import(path, channels=1):
    """
    Import data produced using adc_sampler.c.
    Returns sample period and ndarray with one column per channel.
    Sampled data for each channel, in dimensions NUM_SAMPLES x NUM_CHANNELS.
    """

    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype=np.uint16)
        data = data.reshape((-1, channels))
    return sample_period, data

#sample_period, data = raspi_import('Y2022-M04-D04-H10-M38-S12.bin')

#sample_period *= 1e-6  # change unit to micro seconds

# Generate time axis
#num_of_samples = data.shape[0]  # returns shape of matrix
#t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)

# Generate frequency axis and take FFT
#freq = np.fft.rfftfreq(n=num_of_samples, d=sample_period)
#spectrum = np.fft.rfft(data, axis=0)  # takes FFT of all channels

#dBA_list = []
dBA_dict = {6.3: -85.4, 8: -77.6, 10: -70.4, 12.5: -63.6, 16: -56.4, 20: -50.4, 25: -44.8, 31.5: -39.5, 40: -34.5, 50: -30.3, 63: -26.2, 80: -22.4, 100: -19.1, 125: -16.2, 160: -13.2, 200: -10.8, 250: -8.7, 315: -6.6, 400: -4.8, 500: -3.2, 630: -1.9, 800: -0.8, 1000: 0.0, 1250: 0.6, 1600: 1.0, 2000: 1.2, 2500: 1.3, 3150: 1.2, 4000: 1.0, 5000: 0.6, 6300: -0.1, 8000: -1.1, 10000: -2.5, 12500: -4.3, 16000: -6.7, 20000: -9.3} #inneholder tabellen over generelle dBA verdier

dBA_key_list = list(dBA_dict.keys()) #inneholder alle keys fra dBA_dict

dBA_value_list = list(dBA_dict.values()) #inneholder alle values fra dBA_dict


def dBA(frekvens, spect, dBA_dict): #tar i rfft av signalet
    dBA_vector = []
    temp = 0

    for i in range(0, len(spect)):
        temp = todB_num(spect[i])
        if(frekvens[i] >= 20000):
            temp += -9.3
        else:
            for j in range(0, len(dBA_dict)-1):
                if(dBA_key_list[j] < frekvens[i] < dBA_key_list[j + 1]):
                    temp += (dBA_value_list[j] + dBA_value_list[j + 1])/2
        
        dBA_vector.append(temp)
    
    return dBA_vector


def todB_num(verdi_rfft):
    num_dB = (20*np.log10(np.abs(verdi_rfft)))
   
    return num_dB


def ekvivalentverdi(period, num_samples, verdi_V):
    p = 0
    for i in range(0, len(verdi_V)):
        p2 = (verdi_V[i])**2
        p += p2
   
    L = 10*np.log(p/(len(verdi_V)))
    return L

'########################################################################'
#Båndpass her  --- 

def butter_bandpass(lowcut, highcut, fs, order=8):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=8):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    w, h = freqz(b, a, fs=fs, worN=2000)
    plt.plot(w, abs(h), label="order = %d" % order)
    print('Len h:')
    print(len(abs(h)))
    print(len(w))
    plt.show()
    y = lfilter(b, a, data)
    return y , abs(h)
    #return abs(h)

'########################################################################'

def kalibrering(kalibreringsverdi, frekvens, spect, dBA_dict):
    verdi_kalib = dBA(frekvens, spect, dBA_dict)
    #print(len(verdi_kalib))
    verdi_temp = 0
    kalib_dBA = []
    #Denne løkken kalibrerer
    for f in range(0, len(spect)):
        verdi_temp = verdi_kalib[f] - kalibreringsverdi
        #verdi_temp = np.fft.iffft(verdi_temp)
        kalib_dBA.append(verdi_temp)


    #print(len(kalib_dBA))
    #print(len(freq))
    return kalib_dBA #Verdi i dB

p0 = 20*10^-6
def todB_num(verdi_rfft):
    num_dB = (20*np.log10(np.abs(verdi_rfft)))
   
    return num_dB



def ekvivalentnivå_mv0(måling_data, v0):
    sum = 0
    for i in range(0, len(måling_data)):
        sum += (måling_data[i]/v0)**2
    L = 10*np.log(1/len(måling_data) + sum)
    return float(L)

#Antar
v0 = 0.02

antall_filer = 2
liste_filer = ['Lydfiler/bil.bin', 'Lydfiler/lastebil.bin']
def plott(antall_filer, liste_filer):
    tid = 0
    tid_vec = []
    y_db = 0
    L = 0
    L_eq = []
    dBA_num = 0
    for n in range(0, antall_filer):
        print(liste_filer[n])
        sample_period, data = raspi_import(liste_filer[n])
        num_of_samples = data.shape[0]
        freq = np.fft.rfftfreq(n=num_of_samples, d=sample_period)
        spectrum = np.fft.rfft(data, axis=0) 
        dBA_num = dBA(freq, spectrum, dBA_dict)
        plt.plot(freq, dBA_num)
        plt.show()

        
        #yy, yh = butter_bandpass_filter(x, 200, 800, 31250, order=8)
        #specty= np.fft.rfft(yh, axis=0) 
        T = num_of_samples * sample_period
        L = ekvivalentnivå_mv0(data, v0)
        
        #L_kalib = kalibrering(2, freq, spectrum, dBA_dict)
        tid += 1
        tid_vec.append(tid)

        print(L)
        #a = plt.stem(tid, L)
        L_eq.append(L)
    
    plt.plot(tid_vec, L_eq)
    plt.show()

plott(2, liste_filer) 




#Testing
'''
a = 0.02
f0 = 600.0
x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
x += a * np.cos(2 * np.pi * f0 * t + .11)
x += 0.03 * np.cos(2 * np.pi * 2000 * t)

spectx= np.fft.rfft(x, axis=0) 
x_db = kalibrering(2, freq, spectx, dBA_dict)

plt.subplot(2, 1, 1)
plt.plot(freq, x_db)

yy, yh = butter_bandpass_filter(x, 200, 800, 31250, order=8)
specty= np.fft.rfft(yh, axis=0) 
y_db = kalibrering(2, freq, specty, dBA_dict)

plt.subplot(2, 1, 2)
plt.plot(freq, y_db)

plt.show()
'''




