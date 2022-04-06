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


#plt.subplot(2, 1, 2)
y = dBA(freq, spectrum, dBA_dict)
y2 = todB_vec(spectrum)

plt.subplot(2, 1, 1)
plt.title("dBA spectrum of signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [dBA]")
plt.plot(freq, y) # get the power spectrum


plt.subplot(2, 1, 2)
plt.title("dBA spectrum of signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [dB]")
plt.plot(freq, y2) # get the power spectrum

my_path = os.path.abspath('/Users/mariabolme/Desktop/Elsys/elsys-prosjekt/Nettside/webkurs/elsysapp/static/Bilder/graph1') # Figures out the absolute path for you in case your working directory moves around.

plt.savefig(my_path)

plt.show()



# Filter a noisy signal.
'''
T = 0.05
nsamples = int(T * fs)
t = np.linspace(0, T, nsamples, endpoint=False)
a = 0.02
f0 = 600.0
x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
x += a * np.cos(2 * np.pi * f0 * t + .11)
x += 0.03 * np.cos(2 * np.pi * 2000 * t)
plt.figure(2)
plt.clf()
plt.plot(t, x, label='Noisy signal')

y = butter_bandpass_filter(data, lowcut, highcut, fs, order=6)
plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
plt.xlabel('time (seconds)')
plt.hlines([-a, a], 0, T, linestyles='--')
plt.grid(True)
plt.axis('tight')
plt.legend(loc='upper left')

plt.show()
'''

#Alternativt bp-filter b
#b,a=scipy.signal.butter(N=6, Wn=[0.25, 0.5], btype='band')
#x = scipy.signal.lfilter(b,a,data)
'########################################################################'
#Tester at filteret funker
'''
fs = 40000
x = butter_bandpass_filter(data, 900, 1600, fs, 5)

plt.subplot(2, 1, 1)
plt.plot(t, x)
plt.xlabel('time (seconds)')
plt.grid(True)
plt.axis('tight')
plt.legend(loc='upper left')


plt.subplot(2, 1, 2)
plt.title("Time domain signal")
plt.xlabel("Time [us]")
plt.ylabel("Voltage")
plt.grid(True)
plt.plot(t, data)

plt.show()

freq2 = np.fft.rfftfreq(n=num_of_samples, d=sample_period)
spectrum2 = np.fft.rfft(x, axis=0)
plt.subplot(3, 1, 1)
plt.title("Power spectrum of signal2")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [dB]")
plt.plot(freq2, 20*np.log10(np.abs(spectrum2)))
plt.plot(freq, 20*np.log10(np.abs(spectrum)))
'''
'''
plt.subplot(3, 1, 2)
plt.title("Power spectrum of signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [dB]")
plt.plot(freq, 20*np.log10(np.abs(spectrum)))
'''


'########################################################################'

# Plot the results in two subplots
# NOTICE: This lazily plots the entire matrixes. All the channels will be put into the same plots.
# If you want a single channel, use data[:,n] to get channel n

'''
plt.subplot(3, 1, 1)
plt.title("Power spectrum of signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [dB]")
plt.plot(freq, 20*np.log10(np.abs(spectrum))) # get the power spectrum

plt.subplot(3, 1, 2)
plt.title("Power spectrum of signal2")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [dB]")
plt.plot(freq2, 20*np.log10(np.abs(spectrum2)))

plt.show()
#plt.savefig('fig.png')
'''


#Dataene fra de forskjellige mikrofonene
'''
#mic_1 = scipy.signal.detrend(data[2000:,0])
#mic_2 = scipy.signal.detrend(data[2000:,3])
#mic_3 = scipy.signal.detrend(data[2000:,4])


#Først båndpassfilter over data
#Prøver meg først på noen tusen Hz til å starte med
#order = 6
#fs = 31250
#cutoff = 500

#b, a = butter_lowpass(cutoff, fs, order)

#Plotting freq respons
#w, h = freqz(b, a, worN=8000)
#plt.subplot(2, 1, 1)
#plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
#plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
#plt.axvline(cutoff, color='k')
##plt.xlim(0, 0.5*fs)
#plt.title("Lowpass Filter Frequency Response")
#plt.xlabel('Frequency [Hz]')
#plt.grid()
#plt.show()

#filter_1 = butter_lowpass_filter(mic_1, cutoff, fs, order)
#filter_2 = butter_lowpass_filter(mic_2, cutoff, fs, order)
#filter_3 = butter_lowpass_filter(mic_3, cutoff, fs, order)

#t2 = t[2000:]
#plt.subplot(3, 1, 1)
#plt.plot(t2, mic_1, 'b-', label='data')
#plt.plot(t2, filter_1, 'g-', linewidth=2, label='filtered data')
#plt.xlabel('Time [sec]')
#plt.grid()
#plt.legend()

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

#plt.show()
'''


#så krysskorelasjon mellom dataen og finn største verdi
'''
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
'''

