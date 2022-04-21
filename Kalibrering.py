# import diverse
import numpy as np
import data_analyze_func as daf
import matplotlib.pyplot as plt

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
vv0 = 0.068 #testvariabel
referansenivaa = 75   #referansenivå = målt_verdi_Leq = 75

#kalib_fil = "Opptaksfiler\\Lydfiler-fra-kalibrering-på-lab\\Y2022-M04-D07-H12-M56-S14.bin" #for windows

kalib_fil = "Opptaksfiler/Lydfiler-fra-kalibrering-på-lab/Y2022-M04-D07-H12-M56-S14.bin" #for mac

def finn_v0(kalib_fil, målt_verdi_Leq):
    sample_period, data = daf.raspi_import(kalib_fil)
    num_of_samples = data.shape[0]
    print("###")
    print(len(data))
    print("v0:")
    print(daf.v0)
    print("Ekvivalentniva:")
    ekvi = daf.ekvivalentniva_mv0(data, vv0)
    print(ekvi)
    
    print(len(data))
    # for x in data:
    #     print(x)
    
    # T = num_of_samples  * sample_period
    T = 62500
    sum_V = 0
    sum_datavektor = 0
    # for-løkke
    for i in range(0, T):
        if(data[i] < 791):
            temp = float(-790 + data[i]) 
        else:
            temp = float(data[i]-790)

        sum_V += (temp)**2
        print(temp)
   
    #Finn ut hvilken som gir riktig resultat for v0:    
    # utregnet_v0 = np.sqrt(1/(T * 1/(10**(målt_verdi_Leq/20))) * sum_V) #(målt_verdi_Leq = 75)
    utregnet_v0 = np.sqrt(sum_V/(T*10**(målt_verdi_Leq/10)))
    return utregnet_v0

referanseverdi = finn_v0(kalib_fil, referansenivaa)
print("Utregnet v0:")
print(referanseverdi)
print("Ikke riktig?")



##########PLOTTING#######################
# Generate time axis
    
sample_period, data = daf.raspi_import(kalib_fil)

num_of_samples = data.shape[0]  # returns shape of matrix
t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)

plt.subplot(2, 1, 1)
plt.title("Time domain signal")
plt.xlabel("Time [us]")
plt.ylabel("Voltage")
plt.plot(t, data)

#plt.show()
plt.savefig('TesterInnholdIData')
    
########SLUTT PLOTTING###############


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





