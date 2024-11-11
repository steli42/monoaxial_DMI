#!/usr/bin/env python
# coding: utf-8
import numpy as np

from auxi import find_files as ff
from auxi import export_legend as el
from auxi import c2s
from auxi.hsv import hsv2rgb
from matplotlib.pyplot import cm, colorbar
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import pandas as pd

import copy

import os

plt.rc('text', usetex=True)
plt.rc('text.latex',preamble='\\usepackage{bm,braket}')

ps = ['M2', 'M4', 'M8', 'M16', 'M32', 'M64']
ps = ['M1', 'M2', 'M4', 'M8', 'M16', 'M32', 'M64']
ps = ['M64']
fns = []
[fns.append(f'/Users/andreas/gits/monoaxial_DMI/tdvp/sk/series_lobs_{p}.csv')  for p in ps]
# [fns.append(f'/Users/andreas/gits/monoaxial_stefan/tdvp/ask/series_lobs_{p}.csv') for p in ps]
print(fns)

fig, axs = plt.subplots(2,1, sharex=True)
axs = axs.ravel()

labels = ['M = 1', 'M = 2', 'M = 4', 'M = 8', 'M = 16', 'M = 32', 'M = 64', 'M = 2', 'M = 4', 'M = 8', 'M = 16', 'M = 32', 'M = 64']
for (idfn, fn) in enumerate(fns):
    try:
        data = pd.read_csv(fn)
    except:
        continue

    ts = np.unique(data['t'])

    for i in 'xyz':
        data[i + '_com'] = data[i] * (-1) * (data['S_z']-1/2)
    avs = data.groupby('t').mean().reset_index()
    xcom = - avs['x_com'] / (avs['S_z']-1/2)
    ycom = - avs['y_com'] / (avs['S_z']-1/2)
    axs[0].plot(ts, avs['S_z'], label=labels[idfn])
    axs[1].plot(ts, xcom, label=labels[idfn])
    # plt.plot(ts, ycom)
    comdf = pd.DataFrame()
    comdf['t'] = ts
    comdf['xcom'] = xcom
    comdf['ycom'] = ycom
    comdf.to_csv('com.csv')
axs[0].set_ylabel('$\\braket{\\hat S_{z}}$')
axs[1].set_ylabel('position $x/a$')
axs[1].set_xlabel('dimensionless time $t|J|$')
plt.legend(loc='center left')
plt.savefig('skyrmion_motion.pdf', dpi=600)