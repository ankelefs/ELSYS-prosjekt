# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:02:56 2022

@author: krire
"""

from cProfile import label
from cmath import cos
import math
from turtle import xcor
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import scipy.signal as signal
from scipy.signal import butter, lfilter, freqz

fs = 31250
"""
I denne filen finnes:
* En funksjon for å dele opp en stor binærfil i mange 1-sekund arrays.
* Finne L_eq_total ved å bruke funksjonen ekvivalentniva_mv0() på hvert sekund, og summere opp.
* 
"""


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

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Import data from bin file
sample_period, data = raspi_import('Y2022-M04-D06-H16-M39-S01.bin')


def L_eq_and_fft (sample_period, data):
    
    num_of_samples_in_bin = data.shape[0]  # returns shape of matrix
    
    #Finner hvor mange sekunder det er i binær-filen
    num_of_seconds_in_bin = int(num_of_samples_in_bin/fs)

    #list_of_seconds er et array som inneholder like mange arrays som det
    #er sekunder i data-arrayet. Hvert av disse arrayene inneholder fs=31250 samplinger. 
    list_of_seconds = np.split(data,num_of_seconds_in_bin)
    
   
    #Variabelen holder summen av 10^(spl_second/10)
    sum_spl = 0

    for each_second in list_of_seconds:
        
        num_of_samples_per_second = each_second.shape[0]  # returns shape of matrix
        #NB! BRUKER NAVNET ekvivalentniva med a i stedet for å
        spl_second = ekvivalentniva_mv0(each_second, v0)
        
        sum_spl += 10**(spl_second/10)

    L_eq_total = 10*np.log10(sum_spl/num_of_seconds_in_bin)
    
    return L_eq_total
    


#data = signal.detrend(data, axis=0) # removes DC component for each channel
#sample_period *= 1e-6  # change unit to micro seconds

# Generate time axis
#num_of_samples = data.shape[0]  # returns shape of matrix
#t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)

# Generate frequency axis and take FFT
#freq = np.fft.fftfreq(n=num_of_samples, d=sample_period)
#spectrum = np.fft.fft(data, axis=0)  # takes FFT of all channels



#    #Finner hvor mange sekunder det er i binær-filen
#    num_of_seconds_in_bin = int(num_of_samples/fs)

    #list_of_seconds er et array som inneholder like mange arrays som det
    #er sekunder i data-arrayet. Hvert av disse arrayene inneholder fs=31250 samplinger. 
#    list_of_seconds = np.split(data,num_of_seconds_in_bin)


    #Variabelen holder summen av 10^(spl_second/10)
#    sum_spl = 0

#    for each_second in list_of_seconds:

        #NB! BRUKER NAVNET ekvivalentniva med a i stedet for å
#        spl_second = ekvivalentniva_mv0(each_second, v0)

#        sum_spl += 10**(spl_second/10)

#    L_eq_total = 10*np.log10(sum_spl/num_of_seconds_in_bin)


    
    #Lag en tom liste som man kan putte inn spl_seconds i.
    '''HUSK Å ENDRE np.log til np.log10 i all kode!!'''
    
#     for j in range(fs):
#         print(each_second[j])
#     print("#####")
    
    

# plankton = np.arange(1,31250*20*60+1,1)

# brugde = np.split(plankton, 1200)


# for i in range(31250):
#     print(brugde[1][i])
    


'#########Kode som finner mest fremtredende frekvens i fft-bilde###########'

mostProminentFreq = np.argmax(spectrum) #Variabelen holder posisjonen i arrayet til den frekvensen som er mest fremtredende.
print('Mest fremtredende Frekvens:',freq[mostProminentFreq])
print(mostProminentFreq)
print('Amplitude til mest fremtredende frekvens: ', (20*np.log10(np.abs(2*(spectrum[mostProminentFreq]))))-60, 'dB')

print(num_of_samples)

'########################################################################'


# Plot the results in two subplots
# NOTICE: This lazily plots the entire matrixes. All the channels will be put into the same plots.
# If you want a single channel, use data[:,n] to get channel n
plt.subplot(2, 1, 1)
plt.title("Time domain signal")
plt.xlabel("Time [us]")
plt.ylabel("Voltage")
plt.plot(t, data)

plt.subplot(2, 1, 2)
plt.title("Power spectrum of signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [dB]")
plt.plot(freq, 20*np.log10(np.abs(spectrum))) # get the power spectrum

plt.show()
plt.savefig('fig.png')


#Dataene fra de forskjellige mikrofonene

mic_1 = scipy.signal.detrend(data[2000:,0])
#mic_2 = scipy.signal.detrend(data[2000:,3])
#mic_3 = scipy.signal.detrend(data[2000:,4])


#Først båndpassfilter over data
#Prøver meg først på noen tusen Hz til å starte med
order = 6
cutoff = 500

b, a = butter_lowpass(cutoff, fs, order)

#Plotting freq respons
w, h = freqz(b, a, worN=8000)
plt.subplot(2, 1, 1)
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()
plt.show()

filter_1 = butter_lowpass_filter(mic_1, cutoff, fs, order)
#filter_2 = butter_lowpass_filter(mic_2, cutoff, fs, order)
#filter_3 = butter_lowpass_filter(mic_3, cutoff, fs, order)

t2 = t[2000:]
plt.subplot(3, 1, 1)
plt.plot(t2, mic_1, 'b-', label='data')
plt.plot(t2, filter_1, 'g-', linewidth=2, label='filtered data')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

#plt.subplot(3, 1, 2)
#plt.plot(t2, mic_2, 'b-', label='data')
#plt.plot(t2, filter_2, 'g-', linewidth=2, label='filtered data')
#plt.xlabel('Time [sec]')
#plt.grid()
#plt.legend()

#plt.subplot(3, 1, 3)
#plt.plot(t2, mic_3, 'b-', label='data')
#plt.plot(t2, filter_3, 'g-', linewidth=2, label='filtered data')
#plt.xlabel('Time [sec]')
#plt.grid()
#plt.legend()

plt.show()

