# -*- coding: utf-8 -*-

import os
import numpy as np
import sys
parent_dir = os.path.dirname(os.path.abspath('..'))
sys.path.append(parent_dir)
from vies.lfp.filter import movingaverage, gaussian_smooth

def extract_frequency_bins(spiketimes, bin_size, start, stop, **kwargs):
    '''
    Parameters
    ----------
    bin_time, spike_count = extract_frequency_bins(spiketimes, bin_size, start, stop, **kwargs)
    
    bin_size: bin size in seconds
    start: the point in the recording where the function starts binning in seconds
    stop: the point in the recording where the function stops binning in seconds
    **kwargs: TYPE
        - smooth_method: by default, no smoothing is performed, but 'moving_average' or 'gaussian' smoothing can be performed
        - smooth_window: if smoothing is performed, provide a smoothing window (number of bins) as an integer. Default is 5 bins.
        - smooth_sigma: if gaussian smoothing is performed, provide a sigma value, denoting the narrowness of the filter
    
    Returns
    -------
    bin_time: numpy array of time bins (center of bins) in seconds
    spike_count: numpy array of spike frequency in each time bin, if several templates are selected this will be a 2D array
    '''
    smooth_method = kwargs.get('smooth_method', 'none')
    smooth_window = kwargs.get('smooth_window', 5)
    n_bins = int((stop - start)/bin_size)
    bin_time = np.arange(0,(stop-start), bin_size) + bin_size/2
    
    spiketimes = spiketimes
    spike_count = np.histogram(spiketimes, n_bins, (start, stop))[0] 
    spike_count = spike_count / bin_size
    
    if smooth_method == 'moving_average':
        spike_count = movingaverage(spike_count,smooth_window)                
    elif smooth_method == 'gaussian':
        smooth_sigma = kwargs.get('smooth_sigma', 2)
        spike_count = gaussian_smooth(spike_count,smooth_window, smooth_sigma)                
    elif smooth_method == 'none':
        pass
    
    return bin_time, spike_count