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
plt.rc('text', usetex=True)
plt.rc('text.latex',preamble='\\usepackage{bm}')
fs = 1000
fs = 1070  # 11 X 11
fs = 1070  # 13 X 13
# fs = 1230  # 15 X 15
# fs = 1350  # 17 X 17
# fs = 1690  # 21 x 21
fs = 2450  # 31 x 31
# fs = 3400  # 45 x 15
in_dir = 'out'
mkr = 's'
my_dpi = 300
ffmt = ''
str = "series_lobs"
mpl.rcParams['figure.figsize'] = (fs/my_dpi,fs/my_dpi)

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.neighbors import KDTree

import os

def X(Oi, Oj, Ok):
    return 1 + Oi.dot(Oj) + Oj.dot(Ok) + Ok.dot(Oi)

def Y(Oi, Oj, Ok):
    return Oi.dot(np.cross(Oj, Ok))

fns = np.sort(ff.find_files(f'*{str}.csv', in_dir))

make_averages_csv = True
plot_individuals = True
use_vlad_colorcode = False

nstates = 1
fig, ax = plt.subplots()
for (idfn, fn) in enumerate(fns):
    print(f'{np.round(idfn/len(fns)*100, decimals=2)}%')
    print(fn)
    fn_repl = fn.replace(f'{str}.csv', '')
    out_dir = f"{fn_repl}/series"
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    data = pd.read_csv(fn)

    times = np.unique(data['t'])

    for (idt,time) in enumerate(times):
        print(idt)
        data_slice = data[data['t']==time]
        x, y, Sx, Sy, Sz = data_slice['x'], data_slice['y'], data_slice['S_x'], data_slice['S_y'], data_slice['S_z']
        S = (Sx**2 + Sy**2 + Sz**2)**(1/2)
        Ox, Oy, Oz = Sx/S, Sy/S, Sz/S

        vabs = max(S)

        Sz = Sz.replace(1.0, 0.5)

        # rtp = np.asarray([c2s.to_spherical(*v) for v in zip(Sx,Sy,Sz)])
        # # print(min(abs(vabs-S)/max(vabs-S)))
        # imag = ax.scatter(x, y, cmap='RdBu_r', c=Sz, marker=mkr, s=90, vmin=-vabs, vmax=vabs)
        # ax.quiver(x, y, Sx, Sy, units='xy', width=0.07, scale=vabs, pivot='middle', color='white')

        normdev = abs(vabs-S)
        normdev = normdev/vabs
        print(min(normdev), max(normdev))

        if use_vlad_colorcode:
            # imag = ax.scatter(x, y, cmap='RdBu_r', c=Sz, marker=mkr, s=90, vmin=-vabs, vmax=vabs)
            for (xi,yi,sx,sy,sz,ds) in zip(x,y,Sx,Sy,Sz,normdev):
                sn = (sx**2+sy**2+sz**2)**0.5
                color = hsv2rgb([sx/sn,sy/sn,sz/sn],0,0)
                ax.scatter(xi, yi, c=ds/2, cmap='Greys', marker='s', s=90, vmin=0, vmax=vabs, edgecolor='None')
                imag = ax.scatter(xi, yi, facecolor=color, edgecolor='None', marker='o', s=90*0.1)
                ax.quiver(xi, yi, sx, sy, units='xy', width=0.08, scale=vabs*1, pivot='middle', color=color)
        else:
            imag = ax.scatter(x, y, cmap='RdBu_r', c=Sz, marker=mkr, s=90, vmin=-vabs, vmax=vabs, edgecolors='none')
            ax.quiver(x, y, Sx, Sy, units='xy', width=0.08, scale=vabs*1, pivot='middle', color='white')

        mmy = np.asarray([np.min(y),np.max(y)])
        mmx = np.asarray([np.min(x),np.max(x)])
        ax.set_ylim(1.5*mmy)
        ax.set_xlim(1.5*mmx)
        ax.axis('off')
        ax.set_aspect('equal')

        dtn = f"{idt}".zfill(4)
        plt.savefig(f'{out_dir}/{dtn}mag{ffmt}', pad_inches=0, bbox_inches='tight', dpi=my_dpi)
        ax.cla()