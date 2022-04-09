from cProfile import label
from cmath import cos
import dataclasses
import math
from stat import FILE_ATTRIBUTE_DIRECTORY
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

#VARIABLER
v0 = 0.02
directory = 'Opptaksfiler'
directory_time = 'Opptaksfiler/OpptaksfilerTimer'

antall_filer_time = 0
L = 0
L_eq = []
tid = 0
tid_vec = []
arr = sorted(os.listdir(directory))
arr_time = sorted(os.listdir(directory_time))

dBA_dict = {6.3: -85.4, 8: -77.6, 10: -70.4, 12.5: -63.6, 16: -56.4, 20: -50.4, 25: -44.8, 31.5: -39.5, 40: -34.5, 50: -30.3, 63: -26.2, 80: -22.4, 100: -19.1, 125: -16.2, 160: -13.2, 200: -10.8, 250: -8.7, 315: -6.6, 400: -4.8, 500: -3.2, 630: -1.9, 800: -0.8, 1000: 0.0, 1250: 0.6, 1600: 1.0, 2000: 1.2, 2500: 1.3, 3150: 1.2, 4000: 1.0, 5000: 0.6, 6300: -0.1, 8000: -1.1, 10000: -2.5, 12500: -4.3, 16000: -6.7, 20000: -9.3} #inneholder tabellen over generelle dBA verdier

dBA_key_list = list(dBA_dict.keys()) #inneholder alle keys fra dBA_dict

dBA_value_list = list(dBA_dict.values()) #inneholder alle values fra dBA_dict

fs = 31250 #samplingfrek, må være en satt variabel



#FUNKSJONER

#Åpner fil
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



#Ekvivalentnivå

def ekvivalentniva_mv0(måling_data, v0):
    sum = 0
    for i in range(0, len(måling_data)):
        sum += (måling_data[i]/v0)**2
    L = 10*np.log(1/len(måling_data) + sum)
    return float(L)

def todB_num(verdi_rfft):
    num_dB = (20*np.log10(np.abs(verdi_rfft)))
   
    return num_dB

#dBA funksjon

def dBA(frekvens, spect, dBA_dict): #tar i rfft av signalet
    dBA_vector = []
    temp = 0

    for i in range(0, len(frekvens)):
        temp = todB_num(spect[i])
        if(frekvens[i] >= 20000):
            temp += -9.3
        else:
            for j in range(0, len(dBA_dict)-1):
                if(dBA_key_list[j] < frekvens[i] < dBA_key_list[j + 1]):
                    temp += (dBA_value_list[j] + dBA_value_list[j + 1])/2
        
        dBA_vector.append(temp)
    
    return dBA_vector




def Prominent_freq(sample_period, data):
    freq_dict = {}
    klass_freq = []
    klass_spect = []
    klass_n = []
    num_of_samples_in_bin = data.shape[0]  # returns shape of matrix
    
    #Finner hvor mange sekunder det er i binær-filen
    num_of_5seconds_in_bin = int(num_of_samples_in_bin/(fs*5))
    rest = num_of_samples_in_bin % (fs*5)
    for i in range(0, rest):
        data = np.delete(data, 0)

    #list_of_seconds er et array som inneholder like mange arrays som det
    #er sekunder i data-arrayet. Hvert av disse arrayene inneholder fs=31250 samplinger. 
    list_of_5seconds = np.split(data,num_of_5seconds_in_bin)
    

    for each_5second in list_of_5seconds:
        num_of_samples_5sec = each_5second.shape[0]
        each_5second = signal.detrend(each_5second, axis=0)
        
        spect_5sec = np.fft.rfft(each_5second, axis=0)
        freq = np.fft.rfftfreq(n=num_of_samples_5sec, d=sample_period)
        dBA_temp = dBA(freq, spect_5sec, dBA_dict)
        mostProminent_index = np.argmax(dBA_temp)
        mostProminent_freq = freq[mostProminent_index]
        if(mostProminent_freq in freq_dict):
            freq_dict[mostProminent_freq] += 1
        else:
            freq_dict[mostProminent_freq] = 1
        

        #mostProminent_spect = spect_5sec[mostProminent_index]
        #klass_freq.append(mostProminent_freq)
        #klass_spect.append(mostProminent_spect)
        
    #klass_dBA = dBA(klass_freq, klass_spect, dBA_dict)
    for key in freq_dict:
        klass_freq.append(key)
        klass_n.append(freq_dict.get(key))
    
    return klass_freq, klass_n
    



def plottName(name):
    my_path = os.path.abspath('/Users/mariabolme/Desktop/Elsys/elsys-prosjekt/Nettside/webkurs/elsysapp/static/Frekvensspekter/'+ name)
    return my_path
# Figures out the absolute path for you in case your working directory moves around.




def plot_frekvens():
    antall_filer = 0
    for filename in arr_time:
        if filename.endswith(".bin"):
            antall_filer += 1 
            print(os.path.join("./Opptaksfiler/OpptaksfilerTimer", filename))
            sample_period, data = raspi_import(os.path.join("./Opptaksfiler/OpptaksfilerTimer", filename))
            sample_period *= 1e-6 
            freq, dBA_plott = Prominent_freq(sample_period, data)  
            plt.title("Mest fremtredende frekvens")
            plt.xlabel("Frekvens [Hz]")
            plt.ylabel("Antall")
            for i in range(len(freq)):
                plt.stem(freq[i], dBA_plott[i])    
            with open("fignummer.txt", "r") as file:
                k = file.read()
                j = int(k)
            if j >= 24:
                with open("fignummer.txt", "w") as file:
                    file.write(str(0))
                j = 0        
            j += 1
            plt.savefig(plottName('graph' + str(j)))
            
            continue
        else:
            continue
    with open("fignummer.txt", "w") as file:
        file.write(str(j))

#plot_frekvens()

'ITERERER GJENNOM FILER + LAGRER BILDENE I NETTTSIDEMAPPEN'



def find_ekvivalens():
    Leq_vec = []
    time = []
    t = 0
    for filename in arr:
        if filename.endswith(".bin"):
            print(os.path.join("./Opptaksfiler", filename))
            sample_period, data = raspi_import(os.path.join("./Opptaksfiler", filename))
            sample_period *= 1e-6  # change unit to micro seconds
            
            tempLeq = ekvivalentniva_mv0(data, v0)

            Leq_vec.append(tempLeq)
            t += 0.33
            time.append(t)

            continue
        else:
            continue
    return Leq_vec, time

def plott_ekvivalens():
    L, t = find_ekvivalens()
    plt.clf()
    plt.title("Ekvivalentnivå")
    plt.xlabel("Tid [s]")
    plt.ylabel("L_eq [dB]") #ekvivalent lydnivå
    plt.plot(t, L)
    with open("fignummer.txt", "r") as file:
        k = file.read()
    plt.savefig(os.path.abspath('/Users/mariabolme/Desktop/Elsys/elsys-prosjekt/Nettside/webkurs/elsysapp/static/Ekvivalentnivå/ekvivalentnivå'+ k))
    plt.show()

#plott_ekvivalens()





