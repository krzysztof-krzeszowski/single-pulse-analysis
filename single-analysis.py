#!/usr/bin/env python

import functions as fn
import sys

if len(sys.argv) != 2:
    print('Usage:')
    print('single-analysis.py hdf5_data_file')
    exit()

f = sys.argv[1]

d = fn.read_data(f)
print(d.shape)
