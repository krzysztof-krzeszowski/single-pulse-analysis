#!/usr/bin/env python

import functions as fn
import sys

from pathlib import Path

CONFIG = {
    'WINDOW_SIZE': 500,
    'WINDOW_STEP': 100,
}

if len(sys.argv) != 2:
    print('Usage:')
    print('single-analysis.py hdf5_data_file')
    exit()

f = Path(sys.argv[1])

if not f.is_file():
    exit('File does not exist')

d = fn.read_data(f)

for f in fn.get_mean_profile(d):
    print(f)
