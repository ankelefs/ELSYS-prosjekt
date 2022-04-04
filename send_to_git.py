#! /usr/bin/env python
import time
import os

time.sleep(5)
try:
    os.system('cd /home/pi/Documents/elsys-prosjekt && git add . && git commit -m "Nytt opptak" && git push') # Pusher nytt opptak til git
    print('git push success')
except: 
    print('git push unsuccesful')