#! /usr/bin/env python
import time
import os

time.sleep(5)
try:
    os.system('cd /home/pi/Documents/elsys-prosjekt && git add . && git commit -m "Nytt opptak" && git push') # Pusher nytt opptak til git
    print('git push success')
except: 
    print('git push unsuccesful')

time.sleep(5)

# try:
#     os.system('cd /home/pi/Documents/elsys-prosjekt && rm *.bin') # Sletter alle binærfiler i mappen
#     print('Binary files deleted')
# except:
#     print('Binary files deletion unsuccessful')

# os.system("python3 *adresse til program*")