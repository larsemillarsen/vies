# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 16:54:21 2022

@author: User
"""

import numpy as np
from scipy.io import loadmat

def load_neuronfile(path, srate, channel, gain, inputrange):
    data = loadmat(path)
    data = data['Meting']['adc'][0][0][:,channel-1]*(inputrange/(2**16))
    data = (data / gain) * 1000 # to get mv
    time = len(data)/srate
    time = np.arange(1/srate, time + 1/srate, 1/srate)
    return time, data