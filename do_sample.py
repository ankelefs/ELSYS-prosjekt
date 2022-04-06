#! /usr/bin/env python
from datetime import datetime
import time
import os
import sys

"""
Her ligger funksjonen som 
    * kjører opptak
    
Koden er skrevet skalerbart med intensjon om å importeres i en annen fil.

Scriptet trenger to argumenter etter script-navnet: sampleTime og numSamples
"""

sampleTime = sys.argv[1]
numSamples = sys.argv[2]

def sample(sampleTime = 31250, numSamples = 1, elsysProsjektMappeRPi = '/home/pi/Documents/elsys-prosjekt',  ):    

    for i in range(numSamples):
        now = datetime.now()
        dt_string = now.strftime("Y%Y-M%m-D%d-H%H-M%M-S%S") # YY-mm-dd-H-M-S
        
        os.system(f'cd {elsysProsjektMappeRPi} && sudo ./adc_sampler {sampleTime} {dt_string}.bin')
        
    print('Program ended succesfully')
    
sample(sampleTime, numSamples)