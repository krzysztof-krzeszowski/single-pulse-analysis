import h5py

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
    
