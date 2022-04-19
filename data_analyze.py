#! /usr/bin/env python
import datetime
import buffer_to_first_analysis
import time
import pull_and_remove
import data_analyze_func
import binary_files_treatment

'''
Denne filen skal kjøre alle funksjonene fra data_analyze_func. Programmet begynner med en time_sleep fram til gitt tidspunkt. Da er raspi og server synkronisert.
Henter ny data fra git. Så behandler man data. Sletter så data (bin filene - rådata). Så time_sleep i 6 timer minus hvor lang tid det tok å behandle dataen. Syklys på ny
'''

##### Variabler #####
elsysProsjektMappeServer = '/Users/mariabolme/Desktop/Elsys/elsys-prosjekt'
elsysProsjektOpptaksfilerMappeServer = '/Users/mariabolme/Desktop/Elsys/elsys-prosjekt/Opptaksfiler'
elsysProsjektOpptaksfilerTimerMappeServer = '/Users/mariabolme/Desktop/Elsys/elsys-prosjekt/Opptaksfiler/OpptaksfilerTimer'

hours = 6 # Må være synkront med RPi-opplastinger
time_sleep = hours * 3600 # Antall timer i sekund


##### Funksjoner #####
def analysisRuntimeDuration(time_after_analysis, time_before_analysis):
    # Henter ut differansen i antall sekund fra time_before_analysis og time_after_analysis
    return int((time_after_analysis - time_before_analysis).total_seconds()) 


##### PROGRAMMET #####
if __name__ == '__main__':
    
    # Synkronisering:
    # Antall sekunder til første databehandling = synkronisering mellom enhetene
    sekunder_til_behandlingsstart = buffer_to_first_analysis.synCRON_maximus() 
    #time.sleep(sekunder_til_behandlingsstart)

    while True: # Skal kjøres i uendelig tid
        
        # Synkronisering #2:
        time_before_analysis = datetime.datetime.now()
        
        # Henter nye opptaksfiler:
        #pull_and_remove.pullFromGit(elsysProsjektMappeServer)
        #binary_files_treatment.mergeBinFilesToHour(elsysProsjektOpptaksfilerMappeServer, elsysProsjektOpptaksfilerTimerMappeServer) # Lager nye binærfiler som omfatter all info for hver relevante time (i tillegg til å beholde filene på X ant. min.)
        
        # Behandling av data:
        # Funk for plotting av most prom frek
        data_analyze_func.plot_frekvens()
        # Funk for plotting av ekvniv
        data_analyze_func.plott_ekvivalens()
            
        # Fjerner ferdigbehandlede opptaksfiler:
        #pull_and_remove.removeBinaryFiles(elsysProsjektMappeServer, elsysProsjektOpptaksfilerMappeServer)
        
        # Pause i seks timer minus tiden databehandlingen tok:
        time_after_analysis = datetime.datetime.now()
        pause_time = time_sleep - analysisRuntimeDuration(time_after_analysis, time_before_analysis)
        time.sleep(pause_time)
        print(pause_time)
