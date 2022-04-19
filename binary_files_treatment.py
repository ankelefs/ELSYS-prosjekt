import os

"""
Denne filen inneholder kode som slår sammen alle binærfiler som har med en spesifikk time å gjøre. Koden er skrevet for å importeres i en annen fil.
"""

# elsysProsjektMappeServer = '/Users/mariabolme/Desktop/Elsys/elsys-prosjekt'
# elsysProsjektOpptaksfilerMappeServer = '/Users/mariabolme/Desktop/Elsys/elsys-prosjekt/Opptaksfiler'
# elsysProsjektOpptaksfilerTimerMappeServer = '/Users/mariabolme/Desktop/Elsys/elsys-prosjekt/Opptaksfiler/OpptaksfilerTimer'


def mergeBinFilesToHour(elsysProsjektMappeServer, elsysProsjektOpptaksfilerMappeServer):
    hoursInDay = ["H00", "H01", "H02", "H03", "H04", "H05", "H06", "H07", "H08", "H09", "H10", "H11", "H12", "H13", "H14", "H15", "H16", "H17", "H18", "H19", "H20", "H21", "H22", "H23"] # Oversikt over alle timene i døgnet (med format)
    listOfFilesInFolder = sorted(os.listdir(elsysProsjektOpptaksfilerMappeServer)) # En liste med alle filer i mappen
            
    for hour in hoursInDay:
        globalFile = ""

        for i in range(len(listOfFilesInFolder)):
            try:
                if hour in listOfFilesInFolder[i].split("-"):
                    fileName = hour + '-whole-hour.bin'                     # Nytt filnavn for all info som omhandler den spesifikke timen
                    globalFile = fileName

                    pathToBinFile = elsysProsjektOpptaksfilerMappeServer + "/" + listOfFilesInFolder[i]     # Path til filen som skal leses
                    
                    try:
                        with open(pathToBinFile, "rb") as tempOpen:    # Åpner 
                            byteArr = bytearray(tempOpen.read())    
                    except IOError:
                        print('Error opening the file')  
                            
                    with open(fileName, "ab") as newFile:   # Lager en ny fil
                            newFile.write(byteArr)          # Appender til filen
            except:
                continue
            
        os.system(f'cd {elsysProsjektMappeServer} && mv {globalFile} Opptaksfiler/OpptaksfilerTimer')
        
# mergeBinFilesToHour(elsysProsjektMappeServer, elsysProsjektOpptaksfilerMappeServer)