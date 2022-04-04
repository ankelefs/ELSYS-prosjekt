#! /usr/bin/env python
from datetime import datetime
import time
import os

# twentyMinSamples = 31250*60*20
twentyMinSamples = 31250*5
elsysProsjektMappe = '/home/pi/Documents/elsys-prosjekt'
numSamples = 2

for i in range(numSamples):
    now = datetime.now()
    os.system(fr'cd {elsysProsjektMappe} && sudo ./adc_sampler {twentyMinSamples} {now}.bin')