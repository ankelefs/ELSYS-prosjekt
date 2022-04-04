#! /usr/bin/env python
import time
import os

sleep = 10
sixHours = 3600*6
elsysProsjektMappe = '/Users/mariabolme/Desktop/Elsys/elsys-prosjekt'

while True:
    os.system(f'cd {elsysProsjektMappe} && git pull') # Henter nye lydopptak
    print('git pull success')

    time.sleep(sleep)
    
    os.system(f'cd {elsysProsjektMappe} && python3 raspi_analyze_copy.py') # Analyse
    print('Analysis complete')

    os.system(f'cd {elsysProsjektMappe} && rm *.bin') # Fjerner alle binær-filer
    print('Deletion success')
        
    os.system(f'cd {elsysProsjektMappe} && git add . && git commit -m "Fjerning av gamle lydopptak" && git push') # Fjerner alle binær-filer fra git
    print('Deletion on git success')
 
    time.sleep(sixHours)