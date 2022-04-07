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

#Import data from bin file
sample_period, data = raspi_import('Lydfiler/Lydprøver/34.bin')


#data = signal.detrend(data, axis=0)  # removes DC component for each channel
sample_period *= 1e-6  # change unit to micro seconds

#Generate time axis
num_of_samples = data.shape[0]  # returns shape of matrix
t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)


#plt.plot(t,data)
#plt.show()


# Generate frequency axis and take FFT
freq = np.fft.rfftfreq(n=num_of_samples, d=sample_period)
spectrum = np.fft.rfft(data, axis=0)  # takes FFT of all channels
print(len(freq))
print(len(spectrum))


'#########Kode som finner mest fremtredende frekvens i fft-bilde###########'

mostProminentFreq = np.argmax(spectrum) #Variabelen holder posisjonen i arrayet til den frekvensen som er mest fremtredende.
print('Mest fremtredende Frekvens:',freq[mostProminentFreq])
print(mostProminentFreq)
print('Amplitude til mest fremtredende frekvens: ', (20*np.log10(np.abs(2*(spectrum[mostProminentFreq]))))-60, 'dB')

'########################################################################'


'##########Kode for dBA'

dBA_dict = {6.3: -85.4, 8: -77.6, 10: -70.4, 12.5: -63.6, 16: -56.4, 20: -50.4, 25: -44.8, 31.5: -39.5, 40: -34.5, 50: -30.3, 63: -26.2, 80: -22.4, 100: -19.1, 125: -16.2, 160: -13.2, 200: -10.8, 250: -8.7, 315: -6.6, 400: -4.8, 500: -3.2, 630: -1.9, 800: -0.8, 1000: 0.0, 1250: 0.6, 1600: 1.0, 2000: 1.2, 2500: 1.3, 3150: 1.2, 4000: 1.0, 5000: 0.6, 6300: -0.1, 8000: -1.1, 10000: -2.5, 12500: -4.3, 16000: -6.7, 20000: -9.3} #inneholder tabellen over generelle dBA verdier

dBA_key_list = list(dBA_dict.keys()) #inneholder alle keys fra dBA_dict

dBA_value_list = list(dBA_dict.values()) #inneholder alle values fra dBA_dict

def dBA(frekvens, spect, dBA_dict): #tar i rfft av signalet
    dBA_vector = []
    temp = 0

    for i in range(0, len(freq)):
        temp = todB_num(spect[i])
        if(frekvens[i] >= 20000):
            temp += -9.3
        else:
            for j in range(0, len(dBA_dict)-1):
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
#p0 = 20*10^-6
#Her er det antatt at pa(t) målt volt er utgangspunktet for dB
def ekvivalentverdi(period, num_samples, verdi_V):
    p = 0
    for i in range(0, len(verdi_V)):
        p2 = (verdi_V[i])**2
        p += p2
   
    L = 10*np.log(p/(len(verdi_V)))
    return L


#Her er det antatt at pa(t) er målt i pascal og at i Pa = tall[dB]* 1/94
#Trenger ikke pa
'''
def toPascal(verdi_dB):
    Pa_vec = []

    for i in range(0, len(verdi_dB)):

        Pa_vec.append(20*10**((verdi_dB[i]/p0)))

    return Pa_vec
'''

'''
def ekvivalentverdi2(period, num_samples, verdi_dB):
    p = 0
    verdi = toPascal(verdi_dB)

    for i in range(0, num_samples):
        p2 = (verdi[i]- p0)**2
        p += p2
   
    L = 10*np.log(p/(num_samples))
    return L
'''

#test = ekvivalentverdi(sample_period, num_of_samples, data)
#print('Dette er en test:')
#print(test)

##############################

plt.subplot(2,1,2)
sect_y = np.fft.rfft(y, axis=0)
print(len(np.real(sect_y)))
plt.plot(freq, np.real(sect_y))
#plt.plot(t, y)
#plt.xscale('log')
plt.title('Butterworth filter frequency response_filterert')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
#plt.axvline(100, color='green')
plt.show()

################################

#SLÅ SAMMEN FREKVENSSPEKTERE - tror dette skal slettes etterhvert
#plt.subplot(2, 1, 2)
#plt.title("Frekvensspektere slått sammen")
#plt.xlabel("Frequency [Hz]")
#plt.ylabel("Power [dB]")
#plt.plot(freq, 20*np.log10(np.abs(spectrum)))
#plt.stem(10, ekvivalentverdi(sample_period, data))
#plt.plot(sample_period, ekvivalentverdi(sample_period, data))
#plt.stem(mostProminentFreq, 20*np.log10(np.abs((2*spectrum[mostProminentFreq]))))
'########################################################################'


#tar inn data i tidsdomene og sender ut data i tidsdomene (mulig A-vektet)
def kalibrering(kalibreringsverdi, frekvens, spect, dBA_dict):
    verdi_kalib = dBA(frekvens, spect, dBA_dict)
    #print(len(verdi_kalib))
    verdi_temp = 0
    kalib_dBA = []
    #Denne løkken kalibrerer
    for f in range(0, len(frekvens)):
        verdi_temp = verdi_kalib[f] - kalibreringsverdi
        #verdi_temp = np.fft.iffft(verdi_temp)
        kalib_dBA.append(verdi_temp)


    #print(len(kalib_dBA))
    #print(len(freq))
    return kalib_dBA #Verdi i dB



'########################################################################'

#Disse 3 linjene er unødvendige men brukt til testing
#f_max =  np.argmax(freq)
#print('største f:')
#print(f_max)

fs = 40000 #nyqvist, må være en satt variabel

def enKlasse(verdi_Pa, klasse_freq, verdi_tid):
    bp_filtrert = butter_bandpass_filter(verdi_tid, klasse_freq-30, klasse_freq + 30, fs, 8)
    spect_klasse = np.fft.rfft(bp_filtrert, axis=0) 
    kalb_filtrert = kalibrering(20, freq, spect_klasse, dBA_dict)
    frekvens_niv = ekvivalentverdi(sample_period, num_of_samples, kalb_filtrert)
    #frekvens_niv2 = ekvivalentverdi2(sample_period, num_of_samples, verdi_dB)
    return frekvens_niv

#Disse to linjene brukes kun til testing
#kjøretøy1_freq = 1000
#kjøretøy1_niv = enKlasse(data, kjøretøy1_freq)


#Dictionary med kjøretøysklasse og frekvens i miden av frekvensområdet, denne fyller brukeren inn!!
klasser = {"Bil": 1600, "Lastebil" : 1000}

klassifiserings_freq = []
klassifiserings_niv = []

def klassifisering(klasser, verdi, verdi_tid):

    for key in klasser.keys():
        klasse_niv = enKlasse(verdi, klasser[key], verdi_tid)  #detter er ekvivalentnivålet fra klassens frekvens
        #print(klasse_niv)
        #print(klasser[key])
        klassifiserings_freq.append(klasser[key])
        klassifiserings_niv.append(float(klasse_niv))



    return klassifiserings_freq, klassifiserings_niv


'########################################################################'

#Tester at fun klassifisering funker

#test_kalib = kalibrering(20, freq, spectrum, dBA_dict)
#test_Pa = toPascal(test_kalib)
#test3 = butter_bandpass_filter(data, 2000, 3000, fs, 6)

 


#spect3= np.fft.rfft(test3, axis=0) 
#test_kalib = kalibrering(2, freq, spect3, dBA_dict)

#x, y =klassifisering(klasser, test_kalib, data)


#plt.bar(x, y, color ='maroon', width = 10.0)
#plt.plot(t, test3)
#testtest = kalibrering(2, freq, spectrum, dBA_dict)

#plt.subplot(2, 1, 1)
#plt.plot(t, data)
#plt.plot(freq, testtest)

#plt.subplot(2, 1, 2)
#plt.plot(t, test3)
#plt.plot(freq, test_kalib)
#plt.show()



#Tester at enKlasse kjører og kan plottes:
'''
plt.title("Power spectrum of signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [dB]")
plt.stem(kjøretøy1_freq, kjøretøy1_niv)
plt.show()
'''

'########################################################################'


'########################################################################'

#Diverse som stod der fra før
'''
plt.subplot(2, 1, 2)
plt.title("Power spectrum of signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [dB]")
plt.stem(1100,kjøretøy1_niv)
'''

'''
y = butter_bandpass_filter(data, 900, 1600, fs, 5)
#print(bp_filtrert)

plt.subplot(2, 1, 1)
plt.title("Time domain signal")
plt.xlabel("Time [us]")
plt.ylabel("Voltage")
plt.plot(t, y)
'''

#plt.subplot(2, 1, 2)
'''
y_test = dBA(freq, spectrum, dBA_dict)
y = kalibrering(20, freq, spectrum, dBA_dict) 
y2 = todB_vec(spectrum)
print(len(y_test))
print(len(freq))
print(len(y2))
print(len(y))
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


plt.show()
'''

'''
=======
plt.show()

my_path = os.path.abspath('/Users/mariabolme/Desktop/Elsys/elsys-prosjekt/Nettside/webkurs/elsysapp/static/Bilder') # Figures out the absolute path for you in case your working directory moves around.
my_file = 'graph.png'
plt.savefig(os.path.join(my_path, my_file))
>>>>>>> 60573d5935e6381c55ad3a357da9b52da5c4fd3f

plt.show()

# Filter a noisy signal.
'''
def ekvivalentnivå_mv0(måling_data, v0):
    sum = 0
    for i in range(0, len(måling_data)):
        sum += (måling_data[i]/v0)**2
    L = 10*np.log(1/len(måling_data) + sum)
    return float(L)


#Antar
v0 = 0.02

'''
T = 0.05
nsamples = int(T * fs)
#t = np.linspace(0, T, nsamples, endpoint=False)
a = 0.02
f0 = 600.0
x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
x += a * np.cos(2 * np.pi * f0 * t + .11)
x += 0.03 * np.cos(2 * np.pi * 2000 * t)
'''


data_spect= np.fft.rfft(data, axis=0) #y-akse i frekvensspekter
#x_db = kalibrering(2, freq, spectx, dBA_dict)
x_Leq = ekvivalentnivå_mv0(data, v0)



'''

plt.subplot(2, 1, 1)
plt.title("data")
plt.grid(True)
plt.plot(freq, np.real(data_spect))

til_filter = []
for g in range(0, len(data)):
    til_filter.append(data[g])


y = butter_bandpass_filter(til_filter, 1200, 1600, 31250, order=3)

print('NY test')
print(len(data))
print(len(y))
print(len(t))

#y_Leq = ekvivalentnivå_mv0(y, v0)

specty= np.fft.rfft(y, axis=0) 
#y_db = kalibrering(2, freq, specty, dBA_dict)

plt.subplot(2, 1, 2)
plt.title("y")
plt.xlabel('Hz')
plt.grid(True)
plt.plot(freq, np.real(specty))
#plt.legend(loc='upper left')
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
plt.legend(loc='upper left')ƒ

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






