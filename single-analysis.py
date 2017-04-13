#!/usr/bin/env python

import functions as fn
import matplotlib.pyplot as plt
import sys

from pathlib import Path

CONFIG = {
    'WINDOW_SIZE': 1000,
    'WINDOW_STEP': 300,
}

if len(sys.argv) != 3:
    print('\nUsage:')
    print('single-analysis.py hdf5_data_file output_file_base\n')
    print('output_file_base - all output files will have names that start with this string')
    print('e.g. 0329+54.21cm -> 0329+54.21cm_baseline.dat etc.')
    exit()

f = Path(sys.argv[1])
FILE_PREFIX = sys.argv[2]

if not f.is_file():
    exit('File does not exist')

d = fn.read_data(f)

N_PULSES = d.shape[0]
N_PULSES = 500

sd = fn.get_sd_from_pulses(d[:N_PULSES], width=CONFIG['WINDOW_SIZE'], step=CONFIG['WINDOW_STEP'])

min_sd = sd[:, 0, 0]
max_sd = sd[:, 1, 0]

top = plt.subplot2grid((2, 1), (0, 0))
top.plot(min_sd, label="Min std")
plt.xlabel('Pulse number')
plt.ylabel('Flux [mJy]')
plt.legend()

bottom = plt.subplot2grid((2, 1), (1, 0))
bottom.plot(max_sd, label="Max std")
plt.xlabel('Pulse number')
plt.ylabel('Flux [mJy]')
plt.legend()

plt.savefig(FILE_PREFIX + '_plot_sd.png')
