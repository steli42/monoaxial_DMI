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

df = pd.read_csv('/Users/andreas/gits/monoaxial_DMI/tdvp/out/series_lobs.csv')
df_slice = df[(df['x']==-0.5) & (df['y']==-0.5)]
plt.plot(df_slice['t'], df_slice['S_x'], label='$S_x$')
plt.plot(df_slice['t'], df_slice['S_y'], label='$S_y$')
plt.plot(df_slice['t'], df_slice['S_z'], label='$S_z$')
plt.legend()
plt.show()