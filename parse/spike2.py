import sys
sys.path.insert(1, r'C:\Users\llarsen\OneDrive - ugentbe\python_functions')
#from vies.f_curry import *
import numpy as np
import h5py


def load_spikematfile(path, channel, srate, scalar):
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())[channel-1]
        data=f[keys]
        dt=1/srate
        #print(data.keys())
        spike_times=np.squeeze(data['times'][:])
        spike_waveforms=np.squeeze(data['values'][:])*scalar

        timevector = np.arange(dt, spike_waveforms.shape[0]*dt, dt)
        spike_waveform_times=np.zeros((spike_waveforms.shape[0], spike_waveforms.shape[1]))
        for i in range(spike_waveforms.shape[1]):
            t = timevector + spike_times[i]
            spike_waveform_times[:,i]=t

    return spike_times, spike_waveform_times, spike_waveforms

def load_spikematfile_bins(path, channel, tmin, tmax, binsize=1):
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())[channel-1]
        data=f[keys]
        #print(data.keys())
        spike_times=np.squeeze(data['times'][:])
        bin_time = np.arange(tmin, tmax, binsize) + binsize/2
        bin_count = np.zeros(len(bin_time))
        for i in range(len(bin_count)):
            for k in range(len(spike_times)):
                if spike_times[k] >= bin_time[i] - binsize/2 and spike_times[k] < bin_time[i] + binsize/2:
                    bin_count[i] = bin_count[i] + 1

        bin_count = bin_count / binsize

        return bin_time, bin_count

def extract_seizure_firing(szstart, szstop, lc_data):
    seizurestart = szstart+60
    seizurestop = szstop+60

    LC_spiketimes, LC_waveformtimes, LC_waveforms = load_spikematfile(lc_data, 1, 30000, 1000)
    time_bins, frequency_bins = load_spikematfile_bins(lc_data,1, 0, 180, binsize=0.1)

    overlap = 0.9
    overlapfactor = int(np.round(1/(1-overlap)))
    #print(time_bins)
    start = np.where(np.ceil(time_bins) == np.ceil(seizurestart))[0][0]
    stop = np.where(np.ceil(time_bins) == np.ceil(seizurestop))[0][0]

    LC_seizure_firing = frequency_bins[start:stop]
    LC_baseline_firing = frequency_bins[0:int(119*overlapfactor)]
    LC_post_seizure_firing = frequency_bins[(stop+2):stop+20*overlapfactor]
    rel_LC_seizure_firing=((np.mean(LC_seizure_firing)-np.mean(LC_baseline_firing))/np.mean(LC_baseline_firing))*100
    rel_LC_post_seizure_firing=((np.mean(LC_post_seizure_firing)-np.mean(LC_baseline_firing))/np.mean(LC_baseline_firing))*100

    return rel_LC_seizure_firing, rel_LC_post_seizure_firing
