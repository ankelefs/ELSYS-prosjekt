import raspi_analyze
import numpy as np

def finn_kalibrering(kalib_fil, målt_verdi):

    #Dette er kopiert fra toppen av koden
    sample_period, data = raspi_analyze.raspi_import(kalib_fil)
    num_of_samples = data.shape[0]
    t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)
    freq = np.fft.rfftfreq(n=num_of_samples, d=sample_period)
    spectrum = np.fft.rfft(data, axis=0)

    diff = np.argmax(spectrum) - målt_verdi
    return diff