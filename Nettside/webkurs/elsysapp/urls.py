from django.contrib import admin
from django.urls import path
from .views import index, sensor, chart, diagram, kart #Relativ import av viewsfunksjonen

appname = "elsysapp"
urlpatterns = [
    path('', index, name='index'),
    path('sensor/', sensor, name='sensor'),
    path('chart/', chart, name='chart'),
    path('diagram/', diagram, name='diagram'),
    path('kart/', kart, name='kart')
]

