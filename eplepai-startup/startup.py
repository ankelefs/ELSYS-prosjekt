#! /usr/bin/env python
import time
import os

os.system("cd /home/pi/Documents/elsys-prosjekt")
time.sleep(10) # Vil teste om Raspberry Pi-en klarer en pull-request fra GitHub etter 10 s
os.system("git pull")

<<<<<<< HEAD
print("hei")
=======
# KAN VÆRE AT DENNE FILEN MÅ LAGRES LOKALT PÅ EPLEPAIEN
>>>>>>> 4dccd0a5b154454fb4846c7b694c01d36153bf5b
