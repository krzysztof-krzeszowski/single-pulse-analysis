#!/usr/bin/env python

import functions as fn
import sys

from pathlib import Path

CONFIG = {
    'WINDOW_SIZE': 500,
}

if len(sys.argv) != 2:
    print('Usage:')
    print('single-analysis.py hdf5_data_file')
    exit()

f = Path(sys.argv[1])

if not f.is_file():
    exit('File does not exist')

d = fn.read_data(f)

sd = fn.get_sd_from_pulses(d[:10], width=CONFIG['WINDOW_SIZE'])
print(sd)
