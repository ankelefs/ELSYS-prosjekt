from tokenize import Imagnumber
from django.contrib import admin
from django.urls import path
from .views import index, sensor, chart, diagram, kart, spekter #Relativ import av viewsfunksjonen
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

appname = "elsysapp"
urlpatterns = [
    path('', index, name='index'),
    path('sensor/', sensor, name='sensor'),
    path('chart/', chart, name='chart'),
    path('diagram/', diagram, name='diagram'),
    path('kart/', kart, name='kart'),
    path('spekter/', spekter, name='spekter')
]


urlpatterns += staticfiles_urlpatterns()
