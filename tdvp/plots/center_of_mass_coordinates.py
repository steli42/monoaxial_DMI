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
# [fns.append(f'/Users/andreas/gits/monoaxial_DMI/tdvp/sk/series_lobs_{p}.csv')  for p in ps]
# [fns.append(f'/Users/andreas/gits/monoaxial_stefan/tdvp/ask/series_lobs_{p}.csv') for p in ps]
# fns = ['/Users/andreas/gits/monoaxial_DMI/tdvp/sk1/series_lobs.csv']
fns = []
# fns.append('/Users/andreas/gits/monoaxial_DMI/tdvp/sk2/series_lobs.csv')
# fns.append('/Users/andreas/gits/monoaxial_DMI/tdvp/sk4/series_lobs.csv')
# fns.append('/Users/andreas/gits/monoaxial_DMI/tdvp/sk8/series_lobs.csv')
fns.append('/Users/andreas/gits/monoaxial_DMI/tdvp/sk16/series_lobs.csv')
fns.append('/Users/andreas/gits/monoaxial_DMI/tdvp/sk16_32/series_lobs.csv')
# fns.append('/Users/andreas/gits/monoaxial_DMI/tdvp/sk1_neg/series_lobs.csv')
# fns.append('/Users/andreas/gits/monoaxial_DMI/tdvp/sk2_neg/series_lobs.csv')
print(fns)

gr = (1+np.sqrt(5))/2.0
dpi = 600
w = 3.25
h = 3.25/gr
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(w, h))

labels = ['M = 1', 'M = 2', 'M = 4', 'M = 8', 'M = 16', 'M = 32', 'M = 64', 'M = 2', 'M = 4', 'M = 8', 'M = 16', 'M = 32', 'M = 64']
# labels = ['$s = \\rm sk$', '$s = \\rm ask$']
for (idfn, fn) in enumerate(fns):
    print(fn)
    try:
        data = pd.read_csv(fn)
    except:
        continue

    ts = np.unique(data['t'])

    for i in 'xyz':
        data[i + '_com'] = data[i] * (data['S_z']-1/2)
    avs = data.groupby('t').mean().reset_index()
    xcom = avs['x_com'] / (avs['S_z']-1/2)
    ycom = avs['y_com'] / (avs['S_z']-1/2)
    if '_neg' in fn:
        coef = np.polyfit(-ts,xcom,1)
        poly1d_fn = np.poly1d(coef) 
        ax.plot(-ts, poly1d_fn(-ts), color='gray')
        ax.plot(-ts, xcom, label=labels[idfn])
    else:
        coef = np.polyfit(ts,xcom,1)
        poly1d_fn = np.poly1d(coef) 
        ax.plot(ts, poly1d_fn(ts), color='gray')
        ax.plot(ts, xcom, label=labels[idfn])
    print(coef)
    comdf = pd.DataFrame()
    comdf['t'] = ts
    comdf['xcom'] = xcom
    comdf['ycom'] = ycom
    comdf.to_csv('com.csv')



ax.set_ylabel('$x_s(t)/a$')
ax.set_xlabel('$t|J|$')
ax.set_xlim([0,500])
# ax.set_ylim([-7,7])
plt.legend(loc='center right')
plt.tight_layout()
plt.savefig('skyrmion_motion.png', dpi=dpi)
plt.savefig('skyrmion_motion.pdf', dpi=dpi)