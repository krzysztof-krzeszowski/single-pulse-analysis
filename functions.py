import h5py
import numpy as np

def get_baseline_single(d, position, width):
    """ Get baseline level from single pulse """
    return np.mean(d[position:position + width])

def get_baselines(d, position, width):
    """ Get baselines from all pulses.
    Probably can be much better than this.
    """
    if isinstance(position, int):
        position = np.array(d.shape[0] * [position])
    if isinstance(width, int):
        width = np.array(d.shape[0] * [width])
    return np.array(list(map(get_baseline_single, d, position, width)))

def get_mean_profile(d):
    """ Get mean profile from all pulses

    d - ndarray with single pulse data
    """
    return d.mean(axis=0)

def get_on_pulse_window(mean_profile):
    """ Finds on pulse window """
    FLUX_THRESH = 0.03
    MARGIN = 700

    max_flux = mean_profile.max()
    max_bin = mean_profile.argmax()

    left_on = 0
    right_on = mean_profile.shape[0]

    for i in range(max_bin, 0, -1):
        if mean_profile[i] < FLUX_THRESH * max_flux:
            left_on = i
            break

    for i in range(max_bin, mean_profile.shape[0]):
        if mean_profile[i] < FLUX_THRESH * max_flux:
            right_on = i
            break
        
    return np.max(left_on - MARGIN, 0), min(right_on + MARGIN, mean_profile.shape[0])

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

def read_data(f):
    """ Get ndarray with single pulses

    f - path to hdf5 file
    """
    return h5py.File(f.as_posix())['data'][:]
    
def rolling_window(a, width, step):
    """ Returns 2-d array with slices of of `a` of `width` separated by `step`"""
    return np.array([a[i:i + width] for i in range(0, len(a) - width, step)])

def subtract_baselines(d, position, width):
    """ Subtract baseline from 2-d array of single pulses """
    if isinstance(position, int):
        position = np.array(d.shape[0] * [position])
    if isinstance(width, int):
        width = np.array(d.shape[0] * [width])
    return np.array(list(map(subtract_baseline_single, d, position, width)))

def subtract_baseline_single(d, position, width):
    """ Subtract baseline from single pulse """
    return d - np.mean(d[position:position + width])
