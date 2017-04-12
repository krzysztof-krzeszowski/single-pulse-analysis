import h5py
import numpy as np

def get_sd_from_pulse(d, width):
    """ Get minimum and maximum value of standard deviation from single pulse

    d - 1-d array
    width - width of sd calculation
    """
    sd = np.std(rolling_window(d, width), axis=1)
    return np.array([np.min(sd), np.max(sd)])

def get_sd_from_pulses(d, width):
    """ Get minimum and maximum values of standard deviation for set of single pulses

    d - 2-d array
    width - width of sd calculation
    """
    return np.array([get_sd_from_pulse(r, width) for r in d])

def get_mean_profile(d):
    """ Get mean profile from all pulses

    d - ndarray with single pulse data
    """
    return d.mean(axis=0)

def read_data(f):
    """ Get ndarray with single pulses

    f - path to hdf5 file
    """
    return h5py.File(f.as_posix())['data'][:]
    
def rolling_window(a, window):
    """ http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
