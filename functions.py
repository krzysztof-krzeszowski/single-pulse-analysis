import h5py

def read_data(f):
    return h5py.File(f)['data'][:]
    
