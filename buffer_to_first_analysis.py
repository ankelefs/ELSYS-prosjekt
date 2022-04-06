import datetime

def synCRON_maximus():
    """
    Funksjonen skal hente tidspunktet ved første kjøring for å synkronisere med cron-table-intervallet til RPi-en. Gir ut differansen i sekunder [s] til neste analyse av opptaksdata.
    1) Henter dato og tiden NÅ
    2) Lager datetime-objekter med dagens dato og tidspunkt hvor analyse-programmet skal kjøre
    3) Henter ut differansen i sekunder for hver av tidspunktene nevnt ovenfor
    4) Returnerer den laveste, positive verdien. Dette er antall sekunder programmet må være pauset FØR FØRSTE KJØRING - deretter er det 6 timer sleep-intervall.
    
    Funksjonen skal altså kun kjøre én gang ved oppstart av analysene.
    
    Serveren skal behandle dataen 30 minutter etter at RPi-en har begynt å overføre dataen til git.
    """
    
    now = datetime.datetime.now() # Tiden nå (i dårlig format)
    tidspunkter = [datetime.datetime.combine(now.date(), datetime.time(00, 30, 00)), datetime.datetime.combine(now.date(), datetime.time(6, 30, 00)), datetime.datetime.combine(now.date(), datetime.time(12, 30, 00)), datetime.datetime.combine(now.date(), datetime.time(18, 30, 00))] # Tidspunkt for RPi å sende til git samt. starte nye opptaksdata (de nye opptaksdataene lagres LOKALT)
    bufre = []

    for tidspunkt in tidspunkter:
        # print(tidspunkt)
        delta = tidspunkt - now
        delta_sec = int(delta.total_seconds()) # Gir antall sekunder til tidspunkt (om det er et tidspunkt tidligere enn now er verdien negativ)
        # print(delta.total_seconds())
        
        if delta_sec > 0:
            bufre.append(delta_sec)
    
    return min(bufre)

    