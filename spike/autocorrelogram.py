# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:02:57 2020

@author: llarsen
"""

import numpy as np

def compute_autocorrelogram(data, binsize, window):
  
    spike_times = data
    
    n_bins = int(window/binsize)
    
    bin_count = np.zeros(n_bins)
    bin_center = np.arange(-window/2, window/2, binsize) + binsize/2
    
    counter = 0
    for i in range(len(bin_count)):
        
        low_edge = bin_center[counter] - binsize/2
        up_edge = bin_center[counter] + binsize/2
        
        for k in range(len(spike_times)):
            norm_spiketrain = spike_times - spike_times[k]
            norm_spiketrain = np.delete(norm_spiketrain,[k])
            
            norm_spiketrain = np.delete(norm_spiketrain, np.argwhere(norm_spiketrain<low_edge))
            norm_spiketrain = np.delete(norm_spiketrain, np.argwhere(norm_spiketrain>up_edge))
            
            for n in range(len(norm_spiketrain)): 
                if norm_spiketrain[n] >= low_edge and norm_spiketrain[n] <= up_edge:
                    bin_count[counter] = bin_count[counter] + 1
        
        counter = counter + 1
    
    
    return bin_center, bin_count