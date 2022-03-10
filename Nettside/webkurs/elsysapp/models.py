from django.db import models # Importerer models

class Person(models.Model): #definerer klassen
    name = models.CharField(max_length=20) #Lager et tekstfelt
    date_of_birth = models.DateField() # Lager et datofelt
    height = models.DecimalField(max_digits=4, decimal_places=1) # Lager et desimalfelt
    timestamp = models.DateField(auto_now_add=True) # Lager et datofelt
    nr_children = models.IntegerField() # Lager et heltallsfelt
    
    def find_age(self): 
        #Funksjon som finner alder til personen
        return age
    
    def has_children(self):
        #Funksjon som returnerer true dersom personen har barn
        return bool(self.nr_children)
        
    def __str__(self):
        #Funksjon som skriver ut et person-objekt. 
        return "Name: {}, Birthday: {}, height: {} cm, has {} children.".format(self.name, self.date_of_birth, self.height, self.nr_children)
    
class SensorData(models.Model):
    data = models.CharField(max_length=128)
    sensor_id =  models.IntegerField()
    timestamp = models.DateTimeField(auto_now_add=True) #Har med tidspunkt

    def __str__(self):
        #Funksjon som skriver ut et sensor-objekt. 
        return "Data fra sensor nr. {}: {}".format(self.sensor_id, self.data)