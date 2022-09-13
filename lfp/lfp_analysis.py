# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 16:55:19 2022

@author: User
"""

import numpy as np
from scipy.signal import blackmanharris, detrend

def spectrogram(data, srate, windowlength=1, overlap=0.5, highfreq=100):
    window = windowlength * srate
    overlapfactor = np.around(1/(1-overlap), 2)
    time = np.around((np.arange(0, ((len(data)/srate)-(window/srate) + (window/srate)/overlapfactor), (window/srate)/overlapfactor) + (window/srate)/2), 2)
    #if overlap > 0:
    #    n_windows = int(np.round(((len(data)/window)) * overlapfactor - 1))
    #else:
    #    n_windows = len(data)/window * overlapfactor

    n_windows = len(time)
    w_data = np.zeros((window, n_windows))

    bh_window=blackmanharris(window)
    for i in range(n_windows):
        start = int(window*i*(1-overlap))
        end = int(start + window)
        seg = detrend(data[start:end])*bh_window
        seg = np.absolute(np.fft.fft(seg)) ** 2
        w_data[:,i] = seg

    frequencies = np.arange(0, highfreq + window/srate, window/srate)

    w_data = w_data[0:int(highfreq+1), :]

    return time, frequencies, w_data

def linelenght(data, srate, start, stop):
    # this outputs the linelength of a signal
    # data is a time series
    # srate is the sample rate of the time series
    # start is the relative start of the calculation in seconds
    # stop is the relative stop of the calculation in seconds
    dt = 1/srate
    data = data[start*srate:(stop*srate+1)]
    linelength = np.sqrt(dt**2 + np.diff(data)**2)
    #linelength = np.sqrt(1 + np.diff(data)**2) * dt
    linelength = np.sum(linelength)
    
    return linelength
