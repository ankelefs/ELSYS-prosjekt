#! /usr/bin/env python
import datetime
import buffer_to_first_analysis
import time
import pull_and_remove
import data_analyze_func
import binary_files_treatment
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from PIL import Image, ImageFont, ImageDraw

##### Variabler #####
elsys_prosjekt = '/Users/ankerlefstad/Desktop/sonuscaptura-demo/elsys-prosjekt'
 
mappe_opptaksfiler = '/Users/ankerlefstad/Desktop/sonuscaptura-demo/elsys-prosjekt/Opptaksfiler'

mappe_midlertidig_plassering = '/Users/ankerlefstad/Desktop/sonuscaptura-demo/elsys-prosjekt/Opptaksfiler/Midlertidig-plassering'

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
        arr = sorted(os.listdir(mappe_opptaksfiler))
        # Iterer gjennom alle filene
        for filename in arr:
            if filename.endswith(".bin"):
                # Hent info fra opptaksfil
                sample_period, data = data_analyze_func.raspi_import(os.path.join("./Opptaksfiler/", filename))
                
                num_of_samples = data.shape[0]
                
                sample_period *= 1e-6 
                
                t = np.linspace(start=0, stop=num_of_samples * sample_period, num=num_of_samples)
                
                # Analyser opptaksfil
                tempLeq = data_analyze_func.ekvivalentniva_mv0(data, v0)
                
                # Henter tidspunkt fra filnavnet
                filename_comp = filename.split('-')
        
        # Lag plott
        # Lydopptaket
        plt.plot(t[100:(31250+100)*10], data[100:(31250+100)*10])
        plt.xlim(0, 10)
        plt.xlabel('Tid [s]')
        plt.ylabel('Spenning [mV]')
        plt.title('Utdrag på ti sekund fra en ti minutters støymåling')
        
        # Tiden opptaket ble gjort
        my_image = Image.open("white-background.png")

        image_editable = ImageDraw.Draw(my_image)
        image_editable.text((15,15), f'Tid: {filename_comp[4]}:{filename_comp[5]}', (237, 230, 211))

        my_image.save("fig-info.png")
        
        # plt.text(0, 0, ')
        # plt.text(10, 10, ) f'Mest fremtredende frekvens: {tempLeq}'
        
        # Frekvensspekteret
        # Ingen plotting

        # fig.tight_layout()
        plt.show()
        plt.savefig('fig.png')
    
            
        # Flytt ferdigbehandlede opptaksfiler til Midlertidig-plassering-mappen
        # pull_and_remove.removeBinaryFiles(elsysProsjektMappeServer, elsysProsjektOpptaksfilerMappeServer)
        
        # Pause i ti minutter minus tiden databehandlingen tok
        # time_after_analysis = datetime.datetime.now()
        # pause_time = time_sleep - analysisRuntimeDuration(time_after_analysis, time_before_analysis)
        # print(f'Sekunder til neste kjøring: {pause_time} s')
        # time.sleep(pause_time)
        
