#! /usr/bin/env python
import buffer_to_first_analysis
import time
import pull_and_remove


##### Variabler #####
elsysProsjektMappeServer = '/Users/mariabolme/Desktop/Elsys/elsys-prosjekt'
elsysProsjektMappeOpptaksfilerServer = '/Users/mariabolme/Desktop/Elsys/elsys-prosjekt/Opptaksfiler'

##### Funksjoner #####


##### PROGRAMMET #####
if __name__ == '__main__':
    
    # Synkronisering:
    sekunder_til_behandlingsstart = buffer_to_first_analysis.synCRON_maximus() # Antall sekunder til første databehandling = synkronisering mellom enhetene
    time.sleep(sekunder_til_behandlingsstart)

    while True:
        # synkronisering #2
        
        # Henter nye opptaksfiler:
        pull_and_remove.pullFromGit(elsysProsjektMappeServer)
        
        # for-løkke for filer
            # behandling av data
            
        # Fjerner ferdigbehandlede opptaksfiler:
        pull_and_remove.removeBinaryFiles(elsysProsjektMappeOpptaksfilerServer)
        
        # times.sleep(6 timer - tiden for bahendling ++)