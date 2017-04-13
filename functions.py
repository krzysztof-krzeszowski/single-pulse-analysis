import h5py
import numpy as np

def get_sd_from_pulse(d, width, step):
    """ Get minimum and maximum value of standard deviation from single pulse

    d - 1-d array
    width - width of sd calculation
    """
    sd = np.std(rolling_window(d, width, step), axis=1)
    return np.array([[np.min(sd), sd.argmin() * step], [np.max(sd), sd.argmax() * step]])

def get_sd_from_pulses(d, width, step):
    """ Get minimum and maximum values of standard deviation for set of single pulses

    d - 2-d array
    width - width of sd calculation
    """
    return np.array([get_sd_from_pulse(r, width, step) for r in d])

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
    
def rolling_window(a, width, step):
    """ Returns 2-d array with slices of of `a` of `width` separated by `step`"""
    return np.array([a[i:i + width] for i in range(0, len(a) - width, step)])
