#! /usr/bin/env python
from datetime import datetime
import time
import os

def sample():
    twentyMinSamples = 31250*60*20
    # twentyMinSamples = 31250*5
    elsysProsjektMappe = '/home/pi/Documents/elsys-prosjekt'
    numSamples = 1

    for i in range(numSamples):
        now = datetime.now()
        dt_string = now.strftime("Y%Y-M%m-D%d-H%H-M%M-S%S") # YY-mm-dd-H-M-S
        
        os.system(f'cd {elsysProsjektMappe} && sudo ./adc_sampler {twentyMinSamples} {dt_string}.bin')
        
    print('Program ended succesfully')
    
sample()