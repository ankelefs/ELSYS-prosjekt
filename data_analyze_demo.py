#! /usr/bin/env python
import datetime
import buffer_to_first_analysis
import time
import pull_and_remove
import data_analyze_func
import binary_files_treatment
import os
import matplotlib.pyplot as plt

##### Variabler #####
elsysProsjektMappeServer_path = '/Volumes/GoogleDrive/Min\ disk/Utdanning/Elektronisk\ systemdesign\ -\ prosjekt\ TTT4270\ /Sonus\ Captura/elsys-prosjekt'
elsysProsjektMappeServer= os.path.normpath(elsysProsjektMappeServer_path)

elsysProsjektOpptaksfilerMappeServer_path = '/Volumes/GoogleDrive/Min\ disk/Utdanning/Elektronisk\ systemdesign\ -\ prosjekt\ TTT4270\ /Sonus\ Captura/elsys-prosjekt/Opptaksfiler'
elsysProsjektOpptaksfilerMappeServer= os.path.normpath(elsysProsjektOpptaksfilerMappeServer_path)

elsysProsjektOpptaksfilerTimerMappeServer_path = '/Volumes/GoogleDrive/Min\ disk/Utdanning/Elektronisk\ systemdesign\ -\ prosjekt\ TTT4270\ /Sonus\ Captura/elsys-prosjekt/Opptaksfiler/OpptaksfilerTimer'
elsysProsjektOpptaksfilerTimerMappeServer = elsysProsjektOpptaksfilerMappeServer= os.path.normpath(elsysProsjektOpptaksfilerTimerMappeServer_path)

# Ti minutter
time_sleep = 60*10
# Fra kalibrering
v0 = 0.00770143

##### Funksjoner #####
def analysisRuntimeDuration(time_after_analysis, time_before_analysis):
    # Henter ut differansen i antall sekund fra time_before_analysis og time_after_analysis
    return int((time_after_analysis - time_before_analysis).total_seconds()) 


##### PROGRAMMET #####
if __name__ == '__main__':
    
    ##### Synkronisering #####
    # Ingen synkronisering i demoen. Hard, manuell start av begge program samtidig.

    # Skal kjøres i uendelig tid
    while True: 
        # Synkronisering #2
        time_before_analysis = datetime.datetime.now()
        
        # Henter nye opptaksfiler:
        # pull_and_remove.pullFromGit(elsysProsjektMappeServer)
        
        ##### Analyse av data #####
        # Vil produsere ETT plott med all informasjon som skal vises på hjemmesiden
        
        # Liste med alle filer i mappen
        arr = sorted(os.listdir(elsysProsjektOpptaksfilerMappeServer))
        # Iterer gjennom alle filene
        for filename in arr:
            if filename.endswith(".bin"):
                # Hent info fra opptaksfil
                sample_period, data = data_analyze_func.raspi_import(os.path.join("./Opptaksfiler/OpptaksfilerTimer", filename))
                
                # Analyser opptaksfil
                sample_period *= 1e-6 
                
                freq, dBA_plott = data_analyze_func.Prominent_freq(sample_period, data)
                
                tempLeq = data_analyze_func.ekvivalentniva_mv0(data, v0)
        
        # Ekvivalentnivå
        l_eq = data_analyze_func.ekvivalentniva_mv0()
        
        # Lag plott
        plt.clf() 
        plt.title("Mest fremtredende frekvens")
        plt.xlabel("Frekvens [Hz]")
        plt.ylabel("Antall")
        plt.stem(freq, dBA_plott)
        plt.show()
            
        # Flytt ferdigbehandlede opptaksfiler til Midlertidig-plassering-mappen
        # pull_and_remove.removeBinaryFiles(elsysProsjektMappeServer, elsysProsjektOpptaksfilerMappeServer)
        
        # Pause i ti minutter minus tiden databehandlingen tok
        # time_after_analysis = datetime.datetime.now()
        # pause_time = time_sleep - analysisRuntimeDuration(time_after_analysis, time_before_analysis)
        # print(f'Sekunder til neste kjøring: {pause_time} s')
        # time.sleep(pause_time)
        
