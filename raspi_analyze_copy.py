from cProfile import label
from cmath import cos
from inspect import modulesbyfile
from lzma import FILTER_LZMA2
import math
from turtle import xcor
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import scipy.signal as signal
from scipy.signal import butter, lfilter, freqz
import os


# MÅ LEGGE INN MAPPE-PATH TIL DER HVOR FIGURENE SKAL LAGRES

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


'########################################################################'
#trengs disse filterne?
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

'########################################################################'

# Import data from bin file

sample_period, data = raspi_import('Lydfiler/lastebil.bin')



#data = signal.detrend(data, axis=0)  # removes DC component for each channel
sample_period *= 1e-6  # change unit to micro seconds

# Generate time axis
num_of_samples = data.shape[0]  # returns shape of matrix
t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)



# Generate frequency axis and take FFT
freq = np.fft.rfftfreq(n=num_of_samples, d=sample_period)
spectrum = np.fft.rfft(data, axis=0)  # takes FFT of all channels




#dBA_array = np.array([]) 

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
            for j in range(0, len(dBA_dict)):
                if(dBA_key_list[j] < frekvens[i] < dBA_key_list[j + 1]):
                    temp += (dBA_value_list[j] + dBA_value_list[j + 1])/2
        
        dBA_vector.append(temp)
    
    return dBA_vector




#For å få måling fra spectrum til dB, tar inn data som har blitt rfft

def todB_vec(verdi_rfft):
    verdi_dB = []
    db_tall = 0

    for i in range(0, len(verdi_rfft)):
        dB_tall = (20*np.log10(np.abs(verdi_rfft[i])))  #Hva er dette 2 tallet, er det pga negative frekvenser
        verdi_dB.append(dB_tall)
   
    return verdi_dB


def todB_num(verdi_rfft):
    num_dB = (20*np.log10(np.abs(verdi_rfft)))
   
    return num_dB



'########################################################################'
#Ekvivalentverdi, er gitt i dB
p0 = 20*10^-6
#Her er det antatt at pa(t) målt volt er utgangspunktet for dB
def ekvivalentverdi(period, num_samples, verdi):
    p = 0
    for i in range(0, num_samples):
        p2 = (verdi[i]- p0)**2
        p += p2
   
    L = 10*np.log(p/(num_samples))
    return L


#Her er det antatt at pa(t) er målt i pascal og at i Pa = tall[dB]* 1/94
def toPascal(verdi):
    verdi = todB_vec(verdi)

    for i in range(0, len(verdi)):
        verdi[i] = 20*10**((verdi[i]/p0))

    return verdi

def ekvivalentverdi2(period, num_samples, verdi):
    p = 0
    verdi = toPascal(verdi)

    for i in range(0, num_samples):
        p2 = (verdi[i]- p0)**2
        p += p2
   
    L = 10*np.log(p/(num_samples))
    return L


test = ekvivalentverdi(sample_period, num_of_samples, data)
#print('Dette er en test:')
#print(test)

'########################################################################'
#Båndpass her  --- dette funket ikke, nytt filter lenger ned.

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y



fs = 40000 #nyqvist eller hva det heter, må være en satt variabel

def enKlasse(verdi, klasse_freq):
    bp_filtrert = butter_bandpass_filter(verdi, klasse_freq-300, klasse_freq + 300, fs, 5)
    frekvens_niv = ekvivalentverdi(sample_period, num_of_samples, bp_filtrert)
    return frekvens_niv

#Disse to linjene brukes kun til testing
#kjøretøy1_freq = 1000
#kjøretøy1_niv = enKlasse(data, kjøretøy1_freq)


#Dictionary med kjøretøysklasse og frekvens i miden av frekvensområdet, denne fyller brukeren inn!!
klasser = {"Bil": 1600, "Lastebil" : 1000}

klassifiserings_freq = []
klassifiserings_niv = []

def klassifisering(klasser, verdi):

    for key in klasser.keys():
        klasse_niv = enKlasse(verdi, klasser[key])  #detter er ekvivalentnivålet fra klassens frekvens
        print(klasse_niv)
        klassifiserings_freq.append(klasser[key])
        klassifiserings_niv.append(float(klasse_niv))


    return klassifiserings_freq, klassifiserings_niv



#Kode kalibrering, men usikker om tallet fra kalibrering skal multipliseres eller adderes
#tar inn data i tidsdomene og sender ut data i tidsdomene (mulig A-vektet)
def kalibrering(kalibreringsverdi, freq_vec, verdi):
    verdi_kalib = []
    verdi_temp = 0

    for f in range(0, len(freq_vec)):
        verdi_temp = verdi[f] - kalibreringsverdi
        #få verdien i dbA her
        verdi_temp = np.fft.iffft(verdi_temp)
        verdi_kalib.append(verdi_temp)

   
    return verdi_kalib

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


#plt.subplot(2, 1, 2)
y = dBA(freq, spectrum, dBA_dict)
y2 = todB_vec(spectrum)

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
