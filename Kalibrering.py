# import diverse
import numpy as np
import data_analyze_func as daf
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import scipy.signal as signal

'''
Denne brukes kun til kalibrering av mikrofonen. Brukes ikke til behandlig av data. Den nederste funksjonen skal brukes.
'''

'''
def finn_kalibrering(kalib_fil, målt_verdi):

    #Dette er kopiert fra toppen av koden
    sample_period, data = raspi_analyze.raspi_import(kalib_fil)
    num_of_samples = data.shape[0]
    t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)
    freq = np.fft.rfftfreq(n=num_of_samples, d=sample_period)
    spectrum = np.fft.rfft(data, axis=0)

    diff = np.argmax(spectrum) - målt_verdi
    return diff
'''

vv0 = 0.05586528 #Referansenivå
T = 62500   #Antall samples i 2 sekund lydfil

vv0 = 0.05597

referansenivaa = 75   #referansenivå = målt_verdi_Leq = 75

#kalib_fil = "Opptaksfiler\\Lydfiler-fra-kalibrering-på-lab\\Y2022-M04-D07-H12-M56-S14.bin" #for windows

kalib_fil = "Opptaksfiler/Lydfiler-fra-kalibrering-på-lab/Testing-på-grupperom-etter-lab/Y2022-M04-D07-H17-M24-S47.bin" #for mac

## Ubruktebeholdere
"""sentrertData = []"""
# toSekDataVektor = []

def finn_v0(kalib_fil, målt_verdi_Leq):
    sample_period, data = daf.raspi_import(kalib_fil)
    num_of_samples = data.shape[0]
    data = signal.detrend(data, axis=0)

    # print(len(data))
    # for x in data:
    #     print(x)
    
    # T = num_of_samples  * sample_period
    
    sum_V = 0
    
    # for-løkke
    for i in range(0, T):

        
        ##sum_V inneholder summen av samplingene kvadrert.
        sum_V += (data[i])**2
        
        
        # ####Sentrerer samplingene manuelt. Bruker i stedet signal.detrend()
        # ##Sentrerer samplingene om 0 ved å trekke fra offset.
        # if(data[i] < 771.038656):
        #     temp = float(-771.038656 + data[i]) 
        # else:
        #     temp = float(data[i]-771.038656)
        # # print(temp)
        """sentrertData.append(data[i])"""
        

        if(data[i] < 791):
            temp = float(-790 + data[i]) 
        else:
            temp = float(data[i]-790)

        sum_V += (temp)**2
        #print(temp)
   
    #Finn ut hvilken som gir riktig resultat for v0:    
    # utregnet_v0 = np.sqrt(1/(T * 1/(10**(målt_verdi_Leq/20))) * sum_V) #(målt_verdi_Leq = 75)

    utregnet_v0 = np.sqrt(sum_V/(T*10**(målt_verdi_Leq/10)))
    return utregnet_v0
        
    
    # ##Variabel som holder summen av målingene for å finne offset
    #     toSekDataVektor.append(float(data[i]))
    
    # ##Bruker gjennomsnittet av to sekund sampling for å finne offset
    # offset = sum(sentrertData)/len(sentrertData)    
    # print("Offset:")
    # print(offset)
         
    #Utregnet #(målt_verdi_Leq = 75)

   

referanseverdi = finn_v0(kalib_fil, referansenivaa)
print("Utregnet v0:")
print(referanseverdi)
print("Ikke riktig?")



##########PLOTTING#######################
# Generate time axis
    
sample_period, data = daf.raspi_import(kalib_fil)
num_of_samples = data.shape[0]  # returns shape of matrix
data = signal.detrend(data, axis=0)

t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples) #For å få 10 sekund byttes T ut med num_of_samples. T gjelder for 2 sekund.

# for item in t:
#     print(item)

plt.subplot(2, 1, 1)
plt.title("Time domain signal")
plt.xlabel("Time [us]")
plt.ylabel("Voltage")
plt.plot(t, data)

plt.show()


# plt.savefig('TesterInnholdIData')

plt.savefig('TesterInnholdIData')

    
########SLUTT PLOTTING###############


print("Ekvivalentniva:")
ekvi = daf.ekvivalentniva_mv0(data, vv0)
print(ekvi)


# def finn_kalibrering(kalib_fil, målt_verdi):
#     #Dette er kopiert fra toppen av koden
#     sample_period, data = raspi_analyze.raspi_import(kalib_fil)
#     num_of_samples = data.shape[0]
#     t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)
#     freq = np.fft.rfftfreq(n=num_of_samples, d=sample_period)
#     spectrum = np.fft.rfft(data, axis=0)
#     diff = np.argmax(spectrum) - målt_verdi
#     return diff
# '''

# målt_verdi_Leq = 75

# kalib_fil = "Y2022-M04-D07-H12_M56-S14.bin"

# def finn_v0(kalib_fil, målt_verdi_Leq):
#     sample_period, data = diverse.raspi_import(kalib_fil)
#     num_of_samples = data.shape[0]
#     T = num_of_samples  * sample_period
#     sum_V = 0
#     # for-løkke
#     for i in range(0, len(data)):
#         sum_V += (data[i])**2

#     v0 = np.sqrt(1/(T * 1/(10**(målt_verdi_Leq/20))) * sum_V)
#     return v0





