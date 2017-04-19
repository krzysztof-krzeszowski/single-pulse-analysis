#!/usr/bin/env python

import functions as fn
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys

from pathlib import Path

mpl.rcParams['figure.figsize'] = [9, 5]
mpl.rcParams['savefig.dpi'] = 150

CONFIG = {
    'MEAN_PROFILE_MAGNIFICATION': 10,
    'WINDOW_SIZE': 1000,
    'WINDOW_STEP': 300,
}

def plot_baselines(b):
    """ Plots baseline series """
    plt.plot(b, label='Baseline')
    plt.xlabel('Pulse number')
    plt.ylabel('Flux [mjy]')
    plt.legend()

    plt.savefig(FILE_PREFIX + '_plot_baseline.png')
    plt.close()

def plot_maxima(mean_profile, max_bin, max_flux):
    plt.plot(
        CONFIG['MEAN_PROFILE_MAGNIFICATION'] * mean_profile, 
        label='Mean profile (x%s)' % CONFIG['MEAN_PROFILE_MAGNIFICATION']
    )
    plt.scatter(max_bin, max_flux, label='Max flux')
    plt.xlim(0, len(mean_profile))
    plt.xlabel('Phase bin')
    plt.ylabel('Flux [mJy]')
    plt.legend()

    plt.savefig(FILE_PREFIX + '_plot_maxima.png')
    plt.close()

def plot_mean_profile(d, left, right):
    """ Plots mean profile """
    plt.plot(d, label='Mean profile')
    plt.xlabel('Phase bin')
    plt.ylabel('Flux [mJy]')
    plt.xlim(left, right)
    plt.legend()

    plt.savefig(FILE_PREFIX + '_plot_mean_profile.png')
    plt.close()

def plot_sd(min_sd, max_sd):
    """ Plots minimum and maximum standard deviation series"""

    top = plt.subplot2grid((2, 1), (0, 0))
    top.plot(min_sd, label='Min std')
    plt.xlabel('Pulse number')
    plt.ylabel('Flux [mJy]')
    plt.legend()
    
    bottom = plt.subplot2grid((2, 1), (1, 0))
    bottom.plot(max_sd, label='Max std')
    plt.xlabel('Pulse number')
    plt.ylabel('Flux [mJy]')
    plt.legend()
    
    plt.savefig(FILE_PREFIX + '_plot_sd.png')
    plt.close()


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

pulses = fn.read_data(f)

N_PULSES = pulses.shape[0]
N_PULSES = 500

pulses = pulses[:N_PULSES]

sd = fn.get_sd_from_pulses(pulses, width=CONFIG['WINDOW_SIZE'], step=CONFIG['WINDOW_STEP'])

min_sd = sd[:, 0, 0]
max_sd = sd[:, 1, 0]

plot_sd(min_sd, max_sd)

off_pulse_windows = sd[:, 0, 1].astype(np.int)

baselines = fn.get_baselines(pulses, position=off_pulse_windows, width=CONFIG['WINDOW_SIZE'])

plot_baselines(baselines)

pulses = fn.subtract_baselines(pulses, position=off_pulse_windows, width=CONFIG['WINDOW_SIZE'])

mean_profile = fn.get_mean_profile(pulses)
LEFT, RIGHT = fn.get_on_pulse_window(mean_profile)

plot_mean_profile(mean_profile, LEFT, RIGHT)

# Remove everything but pulse window
pulses = pulses[:, LEFT:RIGHT]
mean_profile = fn.get_mean_profile(pulses)

maxima = fn.get_maxima(pulses)

max_bin, max_flux = maxima.T
del(maxima)

plot_maxima(mean_profile, max_bin, max_flux)
