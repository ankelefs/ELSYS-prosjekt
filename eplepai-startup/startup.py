#! /usr/bin/env python
import time
import os

os.system("cd /home/pi/Documents/elsys-prosjekt")
time.sleep(10) # Vil teste om Raspberry Pi-en klarer en pull-request fra GitHub etter 10 s
os.system("git pull")

# KAN VÆRE AT DENNE FILEN MÅ LAGRES LOKALT PÅ EPLEPAIEN
