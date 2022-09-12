from scipy.signal import butter, filtfilt, decimate, blackmanharris, detrend, gaussian
import numpy as np
from scipy.io import loadmat
import h5py
import math

def butter_lowpass(cutOff, fs, order=2):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low')
    return b, a

def butter_lowpass_filter(data, cutOff, fs, order=2):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_highpass(cutOff, fs, order=2):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='high')
    return b, a

def butter_highpass_filter(data, cutOff, fs, order=2):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_bandstop(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    return b, a

def butter_bandstop_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandstop(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def load_neuronfile(path, srate, channel, gain, inputrange):
    data = loadmat(path)
    data = data['Meting']['adc'][0][0][:,channel-1]*(inputrange/(2**16))
    data = (data / gain) * 1000 # to get mv
    time = len(data)/srate
    time = np.arange(1/srate, time + 1/srate, 1/srate)
    return time, data

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

def movingaverage(values,window):
    weigths = np.repeat(1.0, window)/window
    smas = np.convolve(values, weigths, 'same')
    return smas # as a numpy array

def gaussian_smooth(values,window,sigma):
    norm_sigma = (window/10)*sigma
    print('norm sigma is ' + str(norm_sigma))
    weights = gaussian(window, sigma)
    weights = weights / np.sum(weights)
    smoothed_array = np.convolve(values, weights, 'same')
    return smoothed_array

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


def round_nearest(x, a):
    return round(round(x / a) * a, -int(math.floor(math.log10(a))))

