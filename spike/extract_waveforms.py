# -*- coding: utf-8 -*-

import numpy as np

def extract_waveforms(data, srate, n_channels, spike_times, window, channels):
    """
    stas = extract_waveforms(data, srate, n_channels, spike_times, window, channels)
        data : path to binary data file
        srate : sample rate of data, i.e. samples per second, integer expected
        n_channels : number of channels in file, integer expected
        spike_times : a list of spike_times, can be list of integers or an numpy array - in samples
        window : the width of spike waveforms to be extracted in milliseconds
        channels : list of channels to extract data from, remember 0-indexing
    
        Returns
        -------
        stas : a numpy matrix of snippets
         - axis 0 = snippet timepoints
         - axis 1 = channels
         - axis 2 = spike number
    """    
    raw_data_path = data
    srate = int(srate)
    n_channels = int(n_channels)
    spike_times = spike_times
    N_t = window # width of waveforms in milliseconds
    channels = channels # channels from which spikes will be extracted
    
    data_file = np.memmap(raw_data_path, offset=0, dtype='int16', mode='r')
    n_samples = int(len(data_file)/n_channels)
    data_file = np.memmap(raw_data_path, offset=0, dtype='int16', mode='r', shape=(n_samples, n_channels))
    
    
    N_t = int(N_t * srate * 1e-3)
    
    #print(type(channels), type(spike_times))
    if isinstance(channels, (int, np.uint32)) and isinstance(spike_times, (int, np.uint32)):
        stas = np.zeros((N_t, 1, 1), dtype=np.float32)

    elif isinstance(channels, (int, np.int64)) and isinstance(spike_times, (np.ndarray, list)):
        stas = np.zeros((N_t, 1, len(spike_times)), dtype=np.float32)
        
    elif isinstance(channels, (int, np.uint32)) and isinstance(spike_times, (np.ndarray, list)):
        stas = np.zeros((N_t, 1, len(spike_times)), dtype=np.float32)
        
    elif isinstance(channels, np.ndarray) and isinstance(spike_times, (int, np.uint32)):
        stas = np.zeros((N_t, len(channels), 1), dtype=np.float32)
        
    elif isinstance(channels, (np.ndarray, list)) and isinstance(spike_times, (np.ndarray, list)):
            stas = np.zeros((N_t, len(channels), len(spike_times)), dtype=np.float32)

    duration = N_t
    offset = duration // 2
    
    count = 0
    if isinstance(spike_times, (np.uint32, int)):
        start = spike_times - offset
        stop = start + duration
        local_chunk = np.array(data_file[start:stop, channels]).astype(np.float)
        if isinstance(channels, (np.uint32, int)):
            stas[:,0,count] = local_chunk
        else:
            stas[:,:,count] = local_chunk
    else:
        for time in spike_times:
            start = int(time - offset)
            stop = int(start + duration)
            local_chunk = np.array(data_file[start:stop, channels]).astype(np.float)
            if isinstance(channels, (np.uint32, int)):
                stas[:,0,count] = local_chunk
            elif isinstance(channels, (np.int64, int)):
                stas[:,0,count] = local_chunk                
            else:
                stas[:,:,count] = local_chunk
            count += 1
    
    stas = np.squeeze(stas)    
    data_file = None
    
    return stas