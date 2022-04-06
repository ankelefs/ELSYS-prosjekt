#! /usr/bin/env python
import time
import os

"""
Her ligger funksjonene som:
    * pull-er fra git
    * fjerner binærfilene
    
Koden er skrevet skalerbart med intensjon om å importeres i en annen fil
"""

# elsysProsjektMappeServer = '/Users/mariabolme/Desktop/Elsys/elsys-prosjekt'

def pullFromGit(elsysProsjektMappeServer):
    os.system(f'cd {elsysProsjektMappeServer} && git pull') # Henter nye lydopptak
    print('git pull success')

def removeBinaryFiles(elsysProsjektMappeServer):
    os.system(f'cd {elsysProsjektMappeServer} && rm *.bin') # Fjerner alle binær-filer
    print('Deletion success')
        
    os.system(f'cd {elsysProsjektMappeServer} && git add . && git commit -m "Fjerning av gamle lydopptak" && git pull && git push') # Fjerner alle binær-filer fra git
    print('Deletion on git success')