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
fns = []
fns.append(f'/Users/andreas/gits/monoaxial_DMI/tdvp/out/series_lobs.csv')
fns.append(f'/Users/andreas/gits/monoaxial_DMI/tdvp/sk/vlad_class.csv')
dataMPS = pd.read_csv(fns[0])
dataLLG = pd.read_csv(fns[1], header=None)

ts = np.unique(dataMPS['t'])
for i in 'xyz':
    dataMPS[i + '_com'] = dataMPS[i] * (-1) * (dataMPS['S_z']-1/2)
avs = dataMPS.groupby('t').mean().reset_index()
xcom = - avs['x_com'] / (avs['S_z']-1/2)
ycom = - avs['y_com'] / (avs['S_z']-1/2)

print(dataMPS)
print(dataLLG)

fig, axs = plt.subplots(2,1, sharex=True)
axs = axs.ravel()
axs[0].plot(dataLLG[0], dataLLG[1], label="LLG")
axs[0].plot(ts/7800, xcom, label="MPS")
axs[1].plot(dataLLG[0], dataLLG[2], label="LLG")
axs[1].plot(ts/7800, ycom, label="MPS")

print(xcom)
print(ycom)

axs[0].set_ylim([0,4])
axs[1].set_xlim([0,0.037])

axs[1].set_xlabel('$t$ (arbitrary units)')
axs[0].set_ylabel('$y/a$')
axs[1].set_ylabel('$x/a$')
plt.legend()
# plt.show()
plt.savefig('LLG_vs_MPS.jpeg', dpi=600)