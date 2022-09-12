# -*- coding: utf-8 -*-

import numpy as np

def extract_lfp(data, srate, n_channels, start, stop, channels):
    """
    stas = extract_waveforms(data, srate, n_channels, spike_times, window, channels)
        data : path to binary data file
        srate : sample rate of data, i.e. samples per second, integer expected
        n_channels : number of channels in file, integer expected
        start : start of LFP to be extracted in seconds
        stop : end of LFP to be extracted in seconds
        channels : list of channels to extract data from, remember 0-indexing
    
        Returns
        -------
        lfp : a numpy matrix of snippets
         - axis 0 = lfp timepoints
         - axis 1 = channels
    """    
    raw_data_path = data
    srate = int(srate)
    n_channels = int(n_channels)
    start = int(start*srate)
    stop = int(stop*srate)
    channels = channels # channels from which spikes will be extracted
    
    data_file = np.memmap(raw_data_path, offset=0, dtype='int16', mode='r')
    n_samples = int(len(data_file)/n_channels)
    data_file = np.memmap(raw_data_path, offset=0, dtype='int16', mode='r', shape=(n_samples, n_channels))
    
    #lfp_length = int((stop - start) * srate)

    lfp = np.array(data_file[start:stop, channels]).astype(np.float)*0.000195
 
    data_file = None
    
    return lfp