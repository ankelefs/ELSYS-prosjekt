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
import sys

fs = 31250
v0 = 0.002

# filnavn = str(sys.argv[1])                      ## Run: python LeqTotal.py 'filnavn.ext'
# pathToFile = "Opptaksfiler/Lydfiler-fra-kalibrering-på-lab/" + filnavn

# Filer fra lab (rå)
# fil1 = 'Opptaksfiler/Lydfiler-fra-kalibrering-på-lab/Y2022-M04-D07-H12-M55-S09.bin'
# fil2 = 'Opptaksfiler/Lydfiler-fra-kalibrering-på-lab/Y2022-M04-D07-H12-M55-S31.bin'
# fil3 = 'Opptaksfiler/Lydfiler-fra-kalibrering-på-lab/Y2022-M04-D07-H12-M55-S53.bin'
# fil4 = 'Opptaksfiler/Lydfiler-fra-kalibrering-på-lab/Y2022-M04-D07-H12-M56-S14.bin'

fil1 = 'Y2022-M04-D07-H12-M55-S09.bin'
fil2 = 'Y2022-M04-D07-H12-M55-S31.bin'
fil3 = 'Y2022-M04-D07-H12-M55-S53.bin'
fil4 = 'Y2022-M04-D07-H12-M56-S14.bin'

filer = [fil1, fil2, fil3, fil4]

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

for fil in filer:
    sample_period, data = raspi_import(fil)

    data = signal.detrend(data, axis=0) # removes DC component for each channel
    sample_period *= 1e-6  # change unit to micro seconds

    # Generate time axis
    num_of_samples = data.shape[0]  # returns shape of matrix
    t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)

    # Generate frequency axis and take FFT
    freq = np.fft.fftfreq(n=num_of_samples, d=sample_period)
    spectrum = np.fft.fft(data, axis=0)  # takes FFT of all channels

    # mostProminentFreq = np.argmax(spectrum) #Variabelen holder posisjonen i arrayet til den frekvensen som er mest fremtredende.
    # print('Mest fremtredende Frekvens:',freq[mostProminentFreq])
    # print(mostProminentFreq)
    # print('Amplitude til mest fremtredende frekvens: ', (20*np.log10(np.abs(2*(spectrum[mostProminentFreq]))))-60, 'dB')
    # print(num_of_samples)

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
    # plt.savefig('fig.png')