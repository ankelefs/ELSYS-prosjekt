from django.db import models # Importerer models

class SensorData(models.Model):
    data = models.CharField(max_length=128)
    sensor_id =  models.IntegerField()
    timestamp = models.DateTimeField(auto_now_add=True) #Har med tidspunkt

    def __str__(self):
        #Funksjon som skriver ut et sensor-objekt. 
        return "Data fra sensor nr. {}: {}".format(self.sensor_id, self.data)