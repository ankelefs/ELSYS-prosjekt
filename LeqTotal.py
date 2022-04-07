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
v0 = 0.002

# dBA_dict = {6.3: -85.4, 8: -77.6, 10: -70.4, 12.5: -63.6, 16: -56.4, 20: -50.4, 25: -44.8, 31.5: -39.5, 40: -34.5, 50: -30.3, 63: -26.2, 80: -22.4, 100: -19.1, 125: -16.2, 160: -13.2, 200: -10.8, 250: -8.7, 315: -6.6, 400: -4.8, 500: -3.2, 630: -1.9, 800: -0.8, 1000: 0.0, 1250: 0.6, 1600: 1.0, 2000: 1.2, 2500: 1.3, 3150: 1.2, 4000: 1.0, 5000: 0.6, 6300: -0.1, 8000: -1.1, 10000: -2.5, 12500: -4.3, 16000: -6.7, 20000: -9.3} #inneholder tabellen over generelle dBA verdier

# dBA_key_list = list(dBA_dict.keys()) #inneholder alle keys fra dBA_dict

# dBA_value_list = list(dBA_dict.values()) #inneholder alle values fra dBA_dict

# """
# I denne filen finnes:
# * En funksjon for å dele opp en stor binærfil i mange 1-sekund arrays.
# * Finne L_eq_total ved å bruke funksjonen ekvivalentniva_mv0() på hvert sekund, og summere opp.
# * 
# """


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
sample_period, data = raspi_import('420Hz.bin')

data = signal.detrend(data, axis=0) # removes DC component for each channel
sample_period *= 1e-6  # change unit to micro seconds

# Generate time axis
num_of_samples = data.shape[0]  # returns shape of matrix
t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)

# Generate frequency axis and take FFT
freq = np.fft.fftfreq(n=num_of_samples, d=sample_period)
spectrum = np.fft.fft(data, axis=0)  # takes FFT of all channels


# def todB_num(verdi_rfft):
#     num_dB = (20*np.log10(np.abs(verdi_rfft)))
   
#     return num_dB

# def dBA(frekvens, spect, dBA_dict): #tar i rfft av signalet
#     dBA_vector = []
#     temp = 0

#     for i in range(0, len(frekvens)):
#         temp = todB_num(spect[i])
#         if(frekvens[i] >= 20000):
#             temp += -9.3
#         else:
#             for j in range(0, len(dBA_dict)-1):
#                 if(dBA_key_list[j] < frekvens[i] < dBA_key_list[j + 1]):
#                     temp += (dBA_value_list[j] + dBA_value_list[j + 1])/2
        
#         dBA_vector.append(temp)
    
#     return dBA_vector


# def ekvivalentniva_mv0(måling_data, v0):
#     sum = 0
#     for i in range(0, len(måling_data)):
#         sum += (måling_data[i]/v0)**2
#     L = 10*np.log(1/len(måling_data) + sum)
#     return float(L)


# def Prominent_freq(sample_period, data):

#     klass_freq = []
#     klass_spect = []
#     num_of_samples_in_bin = data.shape[0]  # returns shape of matrix
    
#     #Finner hvor mange sekunder det er i binær-filen
#     num_of_5seconds_in_bin = int(num_of_samples_in_bin/(fs))

#     #list_of_seconds er et array som inneholder like mange arrays som det
#     #er sekunder i data-arrayet. Hvert av disse arrayene inneholder fs=31250 samplinger. 
#     list_of_5seconds = np.split(data,num_of_5seconds_in_bin)
    

#     for each_5second in list_of_5seconds:
#         num_of_samples_5sec = each_5second.shape[0]
#         each_5second = signal.detrend(each_5second, axis=0)
        
#         spect_5sec = np.fft.rfft(each_5second, axis=0)
#         freq = np.fft.rfftfreq(n=num_of_samples_5sec, d=sample_period)
#         dBA_temp = dBA(freq, spect_5sec, dBA_dict)
#         mostProminent_index = np.argmax(dBA_temp)
#         mostProminent_freq = freq[mostProminent_index]
#         mostProminent_spect = spect_5sec[mostProminent_index]
#         klass_freq.append(mostProminent_freq)
#         klass_spect.append(mostProminent_spect)
        
#     klass_dBA = dBA(klass_freq, klass_spect, dBA_dict)
    
#     return klass_freq, klass_dBA


# test = Prominent_freq(sample_period, data)

# print('HEIHEIHEI')
# for element in test:
#     print(element)
# print('HEIHEIHEI')    



# #Finner hvor mange sekunder det er i binær-filen
# num_of_seconds_in_bin = int(num_of_samples/fs)

# #list_of_seconds er et array som inneholder like mange arrays som det
# #er sekunder i data-arrayet. Hvert av disse arrayene inneholder fs=31250 samplinger. 
# list_of_seconds = np.split(data,num_of_seconds_in_bin)


# #Variabelen holder summen av 10^(spl_second/10)
# sum_spl = 0

# for each_second in list_of_seconds:
    
#     #NB! BRUKER NAVNET ekvivalentniva med a i stedet for å
#     spl_second = ekvivalentniva_mv0(each_second, v0)
    
#     sum_spl += 10**(spl_second/10)

# L_eq_total = 10*np.log10(sum_spl/num_of_seconds_in_bin)


    
    #Lag en tom liste som man kan putte inn spl_seconds i.
    #'''HUSK Å ENDRE np.log til np.log10 i all kode!!'''
    
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



#så krysskorelasjon mellom dataen og finn største verdi

# mic_12 = np.correlate(mic_1, mic_2, 'full')
# mic_13 = np.correlate(mic_1, mic_3, 'full')
# mic_23 = np.correlate(mic_2, mic_3, 'full')
# mic_11 = np.correlate(mic_1, mic_1, 'full')

# fig = plt.figure()
# ax1 = fig.add_subplot(411)
# ax1.xcorr(mic_1, mic_2, usevlines=True, maxlags=29)
# ax1.grid(True)

# ax2 = fig.add_subplot(412)
# ax2.xcorr(mic_1, mic_3, usevlines=True, maxlags=29)
# ax2.grid(True)

# ax3 = fig.add_subplot(413)
# ax3.xcorr(mic_2, mic_3, usevlines=True, maxlags=29)
# ax3.grid(True)


# ax3 = fig.add_subplot(414)
# ax3.xcorr(mic_1, mic_1, usevlines=True, maxlags=29)
# ax3.grid(True)

# plt.show()

# max_11 = np.argmax(mic_11)
# max_1 = np.argmax(mic_12) - max_11
# max_2 = np.argmax(mic_13) - max_11
# max_3 = np.argmax(mic_23) - max_11

# print(max_1)
# print(max_2)
# print(max_3)
# print(max_11)

# t_delta_1 = max_1/31250
# t_delta_2 = max_2/31250
# t_delta_3 = max_3/31250

# #mattematisk formel for vinkel basert på matten
# d = 0.055
# c = 343
# #cos(vinkel) = (t_delta * c)/d
# vinkel_1 = math.degrees(np.arccos((t_delta_1*c)/d))
# vinkel_2 = math.degrees(np.arccos((t_delta_2*c)/d))
# vinkel_3 = math.degrees(np.arccos((t_delta_3*c)/d))

# vinkel = math.degrees(np.arctan2(np.sqrt(3)*(max_1 + max_2),(max_1 - max_2 - 2*max_3)))

# print(vinkel_1)
# print(vinkel_2)
# print(vinkel_3)
# print(vinkel)