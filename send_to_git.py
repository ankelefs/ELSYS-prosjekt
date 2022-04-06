#! /usr/bin/env python
import os

def run():
    elsysProsjektMappeRPi = '/home/pi/Documents/elsys-prosjekt'
        
    os.system(f'cd {elsysProsjektMappeRPi} && git add . && git commit -m "Nye opptak (.bin)" && git pull && git push') # Pusher alle nye filer til git, som vil være opptaksfilene i .bin-format. Serveren sletter opptaksfilene etter at de er ferdigbehandlet
        
    print('Program ended succesfully')