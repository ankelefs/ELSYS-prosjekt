from django.shortcuts import render
from django.http import HttpResponse, QueryDict, JsonResponse
from django.middleware.csrf import get_token
from django.views.decorators.csrf import csrf_exempt
import json 
from .models import SensorData


def index(request):
    print("Dette blir printa i terminalen")
    context = {} # Tom dictionary som blir brukt senere!
    return render(request, "elsysapp/index.html", context)


def sensor(request):
    if request.method == "POST":
        data =  QueryDict(request.body) # Gjør data fra request om til en dictionary
        sensor_id = data['sensorID'] # Lagrer sensorIDen til requesten 
        sensor_value = data['sensorData'] # Lagrer sensorverdien til requesten

        #Lager sensorobjektet
        s = SensorData(sensor_id=sensor_id, data=sensor_value)
        s.save()
        return HttpResponse("SUCCESS")
    elif request.method == "GET":
        """Dette MÅ være med!"""
        response = HttpResponse("")
        csrf_token = get_token(request)
        return response


def chart(request):
    labels = [] # Holder navnene på stolpene i stolediagrammet.
    data = []   # Holder høyden til stolpene i diagrammet.

    objects = SensorData.objects    # Queryset som holder alle databaseobjektene.
    
    ids = set()                                                     # Et set er en liste som ikke kan inneholde duplikater.
    [ids.add(id[0]) for id in objects.values_list("sensor_id")]     # List comprehension. Hent ut alle sensor_id fra databasen og legg dem til i ids.
    ids = list(ids)

    counts = [0] * len(ids)
    for i, id in enumerate(ids):
        counts[i] = counts[i] + objects.filter(sensor_id=id).count()

    labels = ["Sensor {}".format(id) for id in ids]
    data = counts

    return JsonResponse(data={
        'labels': labels,
        'data': data,
    })

def diagram(request):
    context = {} # Tom dictionary som blir brukt senere!
    return render(request, "elsysapp/diagram.html", context)

def kart(request):
    context = {} # Tom dictionary som blir brukt senere!
    return render(request, "elsysapp/kart.html", context)

def klokkeslett(request):
    context = {} # Tom dictionary som blir brukt senere!
    return render(request, "elsysapp/klokkeslett.html", context)

def spekter(request):
    context = {}
    return render(request,"elsysapp/spekter.html", context)