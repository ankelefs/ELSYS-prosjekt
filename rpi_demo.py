import datetime
import time
import os


##### Variabler #####
minutes = 2
rpi_wd = '/home/pi/Documents/elsys-prosjekt'
samples = 31250
# Ti minutter sampling
sampling_time = samples * 60 * minutes
time_sleep = (60 * minutes) + 30


##### Funksjoner ####
def sampling():
    print('>>> Sampler ...')
    now = datetime.datetime.now()
    dt_string = now.strftime("Y%Y-M%m-D%d-H%H-M%M-S%S") # YY-mm-dd-H-M-S
    
    os.system(f'cd {rpi_wd} && sudo ./adc_sampler {sampling_time} {dt_string}.bin')
    print('==> Sampling ferdig')
    
def move_to_folder():
    os.system(f'cd {rpi_wd} && mv *.bin Opptaksfiler')
    print('==> Flyttet filer til Opptaksfiler')
    
def send_to_github():
    print('>>> Opplasting til GitHub ...')
    os.system(f'cd {rpi_wd} && git pull && git add . && git commit -m "Nye opptak (.bin)" && git push')
    print('==> Opplasting ferdig')

def analysisRuntimeDuration(time_after_analysis, time_before_analysis):
    # Henter ut differansen i antall sekund fra time_before_analysis og time_after_analysis
    return int((time_after_analysis - time_before_analysis).total_seconds()) 
    
    
##### Programmet #####
if __name__ == '__main__':
    # Synkronisering
    
    while True:
        print('##### KJØRING #####')
        
        time_before = datetime.datetime.now()
        
        sampling()
        move_to_folder()
        send_to_github()
        
        time_after = datetime.datetime.now()
        pause_time = time_sleep - analysisRuntimeDuration(time_after, time_before)
        print(f'==> Sekunder til neste kjøring: {pause_time}')
        time.sleep(pause_time)