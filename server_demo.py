#! /usr/bin/env python
import datetime
import time
import pull_and_remove
import data_analyze_func
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFont, ImageDraw


##### Variabler #####
elsys_prosjekt = '/Users/ankerlefstad/Desktop/sonuscaptura-demo/elsys-prosjekt'
mappe_opptaksfiler = '/Users/ankerlefstad/Desktop/sonuscaptura-demo/elsys-prosjekt/Opptaksfiler'
mappe_midlertidig_plassering = '/Users/ankerlefstad/Desktop/sonuscaptura-demo/elsys-prosjekt/Opptaksfiler/Midlertidig-plassering'
minutes = 2
# Ti minutter
time_sleep = 60 * minutes
# time_sleep = 10
# Fra kalibrering
v0 = 0.00770143


##### Funksjoner #####
def analysisRuntimeDuration(time_after_analysis, time_before_analysis):
    # Henter ut differansen i antall sekund fra time_before_analysis og time_after_analysis
    return int((time_after_analysis - time_before_analysis).total_seconds()) 

def cleanup():
    os.system(f'cd {mappe_opptaksfiler} && mv *.bin Midlertidig-plassering')


##### Programmet #####
if __name__ == '__main__':
    # Synkronisering med RPi
    time.sleep(time_sleep + 60)

    while True: 
        print('##### KJØRING #####')
        
        # Synkronisering #2
        time_before_analysis = datetime.datetime.now()
        
        # Henter nye opptaksfiler:
        print('>>> Henter fra GitHub ...')
        pull_and_remove.pullFromGit(elsys_prosjekt)
        print('==> Hentet fra GitHub')
        
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
                print('>>> Analyserer opptaksfil ...')
                tempLeq = data_analyze_func.ekvivalentniva_mv0(data, v0)
                
                # Henter tidspunkt fra filnavnet
                filename_comp = filename.split('-')
        
        
        ##### Plott #####
        print('>>> Lager plott ...')
        # Lydopptaket
        plt.plot(t[100:(31250+100)*10], data[100:(31250+100)*10], color="#BDD545")
        plt.xlim(0, 10)
        plt.xlabel('Tid [s]')
        plt.ylabel('Spenning [mV]')
        plt.title('Utdrag fra støymålingen')
        plt.tight_layout()
        # plt.show()
        # plt.savefig('HVOR?/fig.png')
        print('==> Plott lagret')
        
        # Annen info
        print('>>> Lager info ...')
        with Image.open("white-background.png") as im:
            draw_im = ImageDraw.Draw(im)
            
            my_font = ImageFont.truetype('/Users/ankerlefstad/Desktop/sonuscaptura-demo/elsys-prosjekt/Roboto/Roboto-Medium.ttf', 60)
            
            # Ikke endre whitespace. Dette er formatet
            draw_im.text((50, 50), f'Ekvivalentnivå siste 5 minutter:   {tempLeq:.2f} dB\nNår målingen ble tatt:                    {filename_comp[3].strip("H")}:{filename_comp[4].strip("M")}:{filename_comp[5].strip("S").split(".")[0]}', fill = (66, 102, 122), font=my_font)
            
            # im.show()
            # im.save('HVOR?/fig-info.png')
            print('==> Info lagret')
            
            
        ##### Avslutning #####
        # Flytt ferdigbehandlede opptaksfiler til Midlertidig-plassering-mappen
        cleanup()
        print('==> Flyttet opptaksfilen')
        
        # Pause i ti minutter minus tiden databehandlingen tok
        time_after_analysis = datetime.datetime.now()
        pause_time = time_sleep - analysisRuntimeDuration(time_after_analysis, time_before_analysis)
        print(f'==> Sekunder til neste kjøring: {pause_time}')
        time.sleep(pause_time)
        
