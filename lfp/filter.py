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
