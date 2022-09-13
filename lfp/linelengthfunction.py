# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:32:46 2020

@author: llarsen
"""

import numpy as np
import scipy.signal as signal
import os
import sys

parent_dir = os.path.dirname(os.path.abspath('..'))
sys.path.append(parent_dir)

from vies.parse.neuron import load_neuronfile
from vies.lfp.filter import butter_bandpass_filter



def linelength_2(path, start, stop, gain, channel):
    # this outputs the linelength of a signal
    # data is a time series
    # srate is the sample rate of the time series
    # start is the relative start of the calculation in seconds
    # stop is the relative stop of the calculation in seconds

    srate = 10000
    channel = int(channel)
    gain = gain
    inputrange = 20

    time, data = load_neuronfile(path, srate, channel, gain, inputrange)

    lowcut_eeg = 1
    desired_eeg_srate = 1000
    highcut_eeg = desired_eeg_srate / 2
    hip_eeg=butter_bandpass_filter(data, lowcut_eeg, highcut_eeg, srate, order=2)
    hip_eeg = signal.decimate(hip_eeg, int(srate/desired_eeg_srate))
    time = np.arange(0,np.max(time),1/desired_eeg_srate)

    dt = 1/srate
    duration = stop - start
    start = int(start*desired_eeg_srate)
    stop = int(stop*desired_eeg_srate+1)

    data = hip_eeg[start:stop]
    #linelength = np.sum(np.sqrt(1 + np.diff(data)**2) * dt) / duration
    diff = np.diff(data)
    #linelength = np.sqrt(1 + np.diff(data)**2) * dt
    linelength = np.sqrt((diff**2) + (dt**2))
    linelength = np.sum(linelength)/duration

    return linelength

#savepath = 'C:\\Users\\llarsen\OneDrive - ugentbe\\52_helpAnirudh\\test.png'
#path = r'\\poweredge\Results3\Anirudh\DG EP stGtACR2\Seizure-Light 4 mW 30 s\AAstGtACR2004L191114A0040.mat'

#test_baseline = linelength_2(path, 0, 20)
#test_stim = linelength_2(path, 20, 30)
#test_seizure = linelength_2(path, 30, 52.3)

#plt.plot(test_baseline)
#plt.show()
