#!/usr/bin/env python

import functions as fn
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import argparse

from pathlib import Path

# matplotlib global options
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
    plt.title(args.prefix)
    plt.legend()

    plt.savefig(args.prefix + '_plot_baseline.png')
    plt.close()

def plot_fluxes_histogram(fluxes):
    plt.hist(fluxes, bins=int(0.05 * N_PULSES), label='Fluxes')
    plt.xlabel('Flux [mJy]')
    plt.ylabel('Counts')
    plt.title(args.prefix)
    plt.legend()

    plt.savefig(args.prefix + '_plot_fluxes_histogram.png')
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
    plt.title(args.prefix)
    plt.legend()

    plt.savefig(args.prefix + '_plot_maxima.png')
    plt.close()

def plot_mean_profile(d, left=None, right=None):
    """ Plots mean profile """
    plt.plot(d, label='Mean profile')
    plt.xlabel('Phase bin')
    plt.ylabel('Flux [mJy]')
    if left and right:
        plt.xlim(left, right)
    plt.title(args.prefix)
    plt.legend()

    if args.m:
        plt.show()
    else:
        plt.savefig(args.prefix + '_plot_mean_profile.png')
    plt.close()

def plot_sd(min_sd, max_sd):
    """ Plots minimum and maximum standard deviation series"""

    top = plt.subplot2grid((2, 1), (0, 0))
    top.plot(min_sd, label='Min std')
    plt.xlabel('Pulse number')
    plt.ylabel('Flux [mJy]')
    plt.title(args.prefix)
    plt.legend()
    
    bottom = plt.subplot2grid((2, 1), (1, 0))
    bottom.plot(max_sd, label='Max std')
    plt.xlabel('Pulse number')
    plt.ylabel('Flux [mJy]')
    plt.title(args.prefix)
    plt.legend()
    
    plt.savefig(args.prefix + '_plot_sd.png')
    plt.close()

parser = argparse.ArgumentParser()
parser.add_argument('file_in', help='HDF5 file containing single pulse data', type=str)
parser.add_argument('-p', '--prefix', help='prefix of output files', type=str, default='out')
parser.add_argument('-w', '--window', help='pulse window', type=int, nargs=2, default=None)
parser.add_argument('-m', help='plot mean profile only', action='store_true')

args = parser.parse_args()

f = Path(args.file_in)

if not f.is_file():
    exit('File does not exist')

pulses = fn.read_data(f)

N_PULSES = pulses.shape[0]
#N_PULSES = 500

pulses = pulses[:N_PULSES]

if args.m:
    mean_profile = fn.get_mean_profile(pulses)
    plot_mean_profile(mean_profile)
    exit()
    

sd = fn.get_sd_from_pulses(pulses, width=CONFIG['WINDOW_SIZE'], step=CONFIG['WINDOW_STEP'])

min_sd = sd[:, 0, 0]
max_sd = sd[:, 1, 0]

plot_sd(min_sd, max_sd)

off_pulse_windows = sd[:, 0, 1].astype(np.int)

baselines = fn.get_baselines(pulses, position=off_pulse_windows, width=CONFIG['WINDOW_SIZE'])

plot_baselines(baselines)

pulses = fn.subtract_baselines(pulses, position=off_pulse_windows, width=CONFIG['WINDOW_SIZE'])

mean_profile = fn.get_mean_profile(pulses)

LEFT, RIGHT = args.window if args.window else fn.get_on_pulse_window(mean_profile)

plot_mean_profile(mean_profile, LEFT, RIGHT)

# Remove everything but pulse window
pulses = pulses[:, LEFT:RIGHT]
mean_profile = fn.get_mean_profile(pulses)

maxima = fn.get_maxima(pulses)

max_bin, max_flux = maxima.T
max_bin = max_bin.astype(np.int)
del(maxima)

plot_maxima(mean_profile, max_bin, max_flux)

fluxes = pulses.sum(axis=1)
plot_fluxes_histogram(fluxes)
