#! /usr/bin/env python
import time
import os

time.sleep(15) # Buffer for å sikre internett-tilgang FØR "git pull"
os.system("cd /home/pi/Documents/elsys-prosjekt")
os.system("git pull")

time.sleep(10) # Buffer før datainnsamling
# os.system("python3 *adresse til program*")