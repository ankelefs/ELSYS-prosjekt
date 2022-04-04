#! /usr/bin/env python
import os

elsysProsjektMappe = '/home/pi/Documents/elsys-prosjekt'    
os.system(f'cd {elsysProsjektMappe} && git add *.bin && git commit -m "Nye opptak (.bin)" && git pull && git push') # Pusher kun .bin-filer til git (serveren sletter de etter behandling)
    
print('Program ended succesfully')