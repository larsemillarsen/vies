# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 15:30:55 2021

@author: Lars
"""

import sys
sys.path.insert(1, r'E:\OneDrive - UGent\python_functions')

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample, hilbert

from math import pi
import math
import OpenEphys as OE

from vies.f_curry import butter_bandpass_filter
from vies.phy_analysis import phy_data
from vies.f_curry import round_nearest


def spike_lfp_radians(filtered_lfp, srate_lfp, lfp_offset, spike_train):
      
    '''
    filtered_lfp = the LFP trace (normally bandpass filtered)
    srate_lfp = srate of LFP
    lfp_offset = offset in LFP recording relative to the time stamps of the spike train
    spike_train = a list of timestamps for the spike train
    
    returns an array of radians, i.e. one radian value per spike, which corresponds to the phase of the LFP at which it occured 
    '''
    srate_lfp = srate_lfp
    dt_lfp = 1/srate_lfp
    precision = str(dt_lfp)
    precision = len(precision.split('.')[1]) 
    
    length_lfp = len(filtered_lfp)/srate_lfp
    t_lfp = np.arange(0, length_lfp, dt_lfp)
    t_lfp = np.around(t_lfp, precision)
    
    #t_lfp_max = lfp_offset + length_lfp
    
    #print(np.max(t_lfp))    
    spike_train = spike_train - lfp_offset
    
    spike_train = spike_train[spike_train > 0]
    spike_train = spike_train[spike_train < length_lfp]

    spike_rounded = np.zeros(len(spike_train))
    index = np.zeros(len(spike_train), dtype=int)    
    
    for i in range(len(spike_train)):
        spike_rounded = round_nearest(spike_train[i], dt_lfp)    
        
        #index=int(spike_rounded/dt_lfp)
        index[i] = int(spike_rounded/dt_lfp)    

    phase_lfp = np.angle(hilbert(filtered_lfp))
    #phase_lfp = np.degrees(phase_lfp+pi)
    #print(len())
    phases = np.zeros(len(spike_train))
    counter = 0
    for i in index:
        phases[counter] = phase_lfp[i]
        counter = counter + 1
     
    radians = phases + pi
    
    return radians

