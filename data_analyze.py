#! /usr/bin/env python
import datetime
import buffer_to_first_analysis
import time
import pull_and_remove
import data_analyze_func


##### Variabler #####
elsysProsjektMappeServer = '/Users/mariabolme/Desktop/Elsys/elsys-prosjekt'
elsysProsjektOpptaksfilerMappeServer = '/Users/mariabolme/Desktop/Elsys/elsys-prosjekt/Opptaksfiler'

hours = 6 # Må være synkront med RPi-opplastinger
time_sleep = hours * 3600 # Antall timer i sekund


##### Funksjoner #####
def analysisRuntimeDuration(time_after_analysis, time_before_analysis):
     return int((time_after_analysis - time_before_analysis).total_seconds()) # Henter ut differansen i antall sekund fra time_before_analysis og time_after_analysis


##### PROGRAMMET #####
if __name__ == '__main__':
    
    # Synkronisering:
    sekunder_til_behandlingsstart = buffer_to_first_analysis.synCRON_maximus() # Antall sekunder til første databehandling = synkronisering mellom enhetene
    time.sleep(sekunder_til_behandlingsstart)

    while True: # Skal kjøres i uendelig tid
        
        # Synkronisering #2:
        time_before_analysis = datetime.datetime.now()
        
        # Henter nye opptaksfiler:
        pull_and_remove.pullFromGit(elsysProsjektMappeServer)
        
        # for-løkke for alle binærfilene i Opptaksfiler-mappen:
            # Behandling av data:
            
        # Fjerner ferdigbehandlede opptaksfiler:
        pull_and_remove.removeBinaryFiles(elsysProsjektMappeServer, elsysProsjektOpptaksfilerMappeServer)
        
        # Pause i seks timer minus tiden databehandlingen tok:
        time_after_analysis = datetime.datetime.now()
        time.sleep(time_sleep - analysisRuntimeDuration(time_after_analysis, time_before_analysis))
