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
from scipy.signal import butter, lfilter, freqz, filtfilt, sosfilt

'''
Denne koden iterer gjennom alle lydfilene som finnes i en mappe og lagrer plottene og sender til nettsiden. 
'''


def plottName(name):
    my_path = os.path.abspath('/Users/mariabolme/Desktop/Elsys/elsys-prosjekt/Nettside/webkurs/elsysapp/static/Frekvensspekter/'+ name)
    return my_path
# Figures out the absolute path for you in case your working directory moves around.


'ITERERER GJENNOM FILER + LAGRER BILDENE I NETTTSIDEMAPPEN'

v0 = 0.02
directory = 'Lydfiler/Lydprøver/'
antall_filer = 0
L = 0
L_eq = []
tid = 0
tid_vec = []
arr = sorted(os.listdir(directory))



def lagre():
    with open("fignummer.txt", "r") as file:
        k = file.read()
        j = int(k)
    if j >= 24:
        with open("fignummer.txt", "w") as file:
            file.write(str(0))
        j = 0
    
    if antall_filer == 3:
        j += 1
        plt.savefig(plottName('graph'+ str(j)))
        plt.clf()
    if antall_filer == 6:
        j += 1
        plt.savefig(plottName('graph' + str(j)))
        plt.clf()
    if antall_filer == 9:
        j += 1
        plt.savefig(plottName('graph' + str(j)))
        plt.clf()
    if antall_filer == 12:
        j += 1
        plt.savefig(plottName('graph' + str(j)))
        plt.clf()
    if antall_filer == 15:
        j += 1
        plt.savefig(plottName('graph' + str(j)))
        plt.clf()
    if antall_filer == 18:
        j += 1
        plt.savefig(plottName('graph' + str(j)))
        plt.clf()
    with open("fignummer.txt", "w") as file:
        file.write(str(j))



for filename in arr:
    if filename.endswith(".bin"): 
        y_db = 0
        dBA_num = 0
        antall_filer += 1
        print(os.path.join("./Lydfiler/Lydprøver", filename))
        sample_period, data = raspi_import(os.path.join("./Lydfiler/Lydprøver", filename))
        num_of_samples = data.shape[0]
        freq = np.fft.rfftfreq(n=num_of_samples, d=sample_period)
        spectrum = np.fft.rfft(data, axis=0) 
        dBA_num = dBA(freq, spectrum, dBA_dict)

        if antall_filer == 1 or antall_filer == 2 or antall_filer == 3:
            plt.title("dBA spectrum of signal")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Power [dBA]")
            plt.plot(freq, dBA_num, color='black')
            lagre()
        if antall_filer == 4 or antall_filer == 5 or antall_filer == 6:
            plt.title("dBA spectrum of signal")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Power [dBA]")
            plt.plot(freq, dBA_num, color='blue')
            lagre()
        if antall_filer == 7 or antall_filer == 8 or antall_filer == 9:
            plt.title("dBA spectrum of signal")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Power [dBA]")
            plt.plot(freq, dBA_num, color='red')
            lagre()
        if antall_filer == 10 or antall_filer == 11 or antall_filer == 12:
            plt.title("dBA spectrum of signal")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Power [dBA]")
            plt.plot(freq, dBA_num, color='green')
            lagre()
        if antall_filer == 13 or antall_filer == 14 or antall_filer == 15:
            plt.title("dBA spectrum of signal")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Power [dBA]")
            plt.plot(freq, dBA_num, color='yellow')
            lagre()
        if antall_filer == 16 or antall_filer == 17 or antall_filer == 18:
            plt.title("dBA spectrum of signal")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Power [dBA]")
            plt.plot(freq, dBA_num, color='pink')
            lagre()
        
        T = num_of_samples * sample_period
        L = ekvivalentnivå_mv0(data, v0)
        
       # L_kalib = kalibrering(2, freq, spectrum, dBA_dict)
        tid += 0.3
        tid_vec.append(tid)

        
        #a = plt.stem(tid, L)
        L_eq.append(L)
        
        continue
        
    else:
        continue
    

plt.clf()
plt.plot(tid_vec, L_eq)
with open("fignummer.txt", "r") as file:
        k = file.read()
plt.savefig(os.path.abspath('/Users/mariabolme/Desktop/Elsys/elsys-prosjekt/Nettside/webkurs/elsysapp/static/Ekvivalentnivå/ekvivalentnivå'+ k))
plt.show()




