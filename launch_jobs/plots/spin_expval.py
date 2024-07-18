#!/usr/bin/env python
# coding: utf-8
import numpy as np

from aux import find_files as ff
from aux import export_legend as el
from aux import c2s
from matplotlib.pyplot import cm, colorbar
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('text.latex',preamble='\\usepackage{bm}')
fs = 1390
in_dir = 'launch_jobs/out'
mkr = 's'
my_dpi = 300
ffmt = ''
str = "/state_mps"
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

nstates = 1
for (idfn, fn) in enumerate(fns):
    fig, ax = plt.subplots()
    print(f'{np.round(idfn/len(fns)*100, decimals=2)}%')
    print(fn)
    fn_repl = fn.replace(f'{str}.csv', '')
    out_dir = fn_repl
    # if os.path.isfile(f'{out_dir}/magnetic_texture{ffmt}'):
    #         continue
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    data = pd.read_csv(fn)

    if (not('approximation' in fn) and os.path.isfile(f'{fn_repl}/energies.csv')):
        data_energies = pd.read_csv(f'{fn_repl}/energies.csv')

    x, y, Sx, Sy, Sz = data['x'], data['y'], data['S_x'], data['S_y'], data['S_z']
    S = (Sx**2 + Sy**2 + Sz**2)**(1/2)
    Ox, Oy, Oz = Sx/S, Sy/S, Sz/S

    vabs = max(S)

    Sz = Sz.replace(1.0, 0.5)
    ax.cla()
    # for ly in np.unique(y)[[1]]:
    #     data_slice = data[data['y'] == ly]
    #     x, y, Sx, Sy, Sz = data_slice['x'], data_slice['y'], data_slice['S_x'], data_slice['S_y'], data_slice['S_z']
    #     S = (Sx**2 + Sy**2 + Sz**2)**(1/2)
    #     imag = ax.plot(x, S)

    rtp = np.asarray([c2s.to_spherical(*v) for v in zip(Sx,Sy,Sz)])
    # print(min(abs(vabs-S)/max(vabs-S)))
    imag = ax.scatter(x, y, cmap='RdBu_r', c=Sz, marker=mkr, s=90, vmin=-vabs, vmax=vabs)
    ax.quiver(x, y, Sx, Sy, units='xy', width=0.07, scale=vabs, pivot='middle', color='white')
    mmy = np.asarray([np.min(y),np.max(y)])
    mmx = np.asarray([np.min(x),np.max(x)])
    ax.set_ylim(1.5*mmy)
    ax.set_xlim(1.5*mmx)
    ax.axis('off')
    ax.set_aspect('equal')

    # print(Ox**2 + Oy**2 + Oz**2)
    # Ox, Oy, Oz = Sx, Sy, Sz
    # print(Ox, Oy, Oz)
    axins = inset_axes(
        ax,
        width="2%",  # width: 5% of parent_bbox width
        height="75%",  # height: 50%
        loc="center right",
        bbox_to_anchor=(0., 0, 0.92, 1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )
    cbar = fig.colorbar(imag, cax=axins, orientation = 'vertical')
    cbar.ax.set_title('$\\langle{\\hat S_{i,z}}\\rangle$')
    # ax.plot([-10,10], [0,0], color='black')
    plt.savefig(f'{out_dir}/mag{ffmt}', pad_inches=0, bbox_inches='tight', dpi=my_dpi)

    fig, ax = plt.subplots()

    Sz = Sz.replace(1.0, 0.5)
    # for ly in np.unique(y)[[1]]:
    #     data_slice = data[data['y'] == ly]
    #     x, y, Sx, Sy, Sz = data_slice['x'], data_slice['y'], data_slice['S_x'], data_slice['S_y'], data_slice['S_z']
    #     S = (Sx**2 + Sy**2 + Sz**2)**(1/2)
    #     imag = ax.plot(x, S)
    # imag = ax.scatter(x, y, cmap='Greys', c=S, marker='h', s=90, vmin=0, vmax=0.5, edgecolors='face')
    print(min(S), max(S), vabs)
    # imag = ax.scatter(x, y, cmap='Greys', c=(vabs-S)/vabs, marker='h', s=90, edgecolors='face', vmin=0,vmax=max(S))
    normdev = abs(vabs-S)
    normdev = normdev/vabs
    print(min(normdev), max(normdev))
    nmax = 1.0
    # normdev = normdev/0.01674
    # normdev = normdev/max(normdev)
    print(min(normdev), max(normdev))
    imag = ax.scatter(x, y, cmap='Greys', c=normdev, marker=mkr, s=90, vmin=0, vmax=nmax, edgecolors='face')
    # ax.quiver(x, y, Sx, Sy, units='xy', width=0.07, scale=vabs, pivot='middle', color='white')
    mmy = np.asarray([np.min(y),np.max(y)])
    mmx = np.asarray([np.min(x),np.max(x)])
    ax.set_ylim(1.5*mmy)
    ax.set_xlim(1.5*mmx)
    ax.axis('off')
    ax.set_aspect('equal')

    # print(Ox**2 + Oy**2 + Oz**2)
    # Ox, Oy, Oz = Sx, Sy, Sz
    # print(Ox, Oy, Oz)
    axins = inset_axes(
        ax,
        width="2%",  # width: 5% of parent_bbox width
        height="75%",  # height: 50%
        loc="center right",
        bbox_to_anchor=(0., 0, 0.92, 1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )
    cbar = fig.colorbar(imag, cax=axins, orientation = 'vertical')
    cbar.ax.set_title('$\\delta s$')
    plt.savefig(f'{out_dir}/norm{ffmt}', pad_inches=0, bbox_inches='tight', dpi=my_dpi)

    # plt.close()
    # plt.plot()
    # el.export_legend(ax, f'{in_dir}/plot_legend{ffmt}')

    # continue

    latt = np.asarray(data[['x', 'y', 'z']])
    kdt = KDTree(latt, leaf_size=30, metric='euclidean')
    onsite = kdt.query_radius(latt, r=0)
    neighbors = kdt.query_radius(latt, r=1.001)
    nns_list = [np.setdiff1d(nn, os) for (os, nn) in zip(onsite, neighbors)]
    # print(nns_list)
    # for (id,nns) in enumerate(nns_list):
    #     for nn in nns:
    #         plt.quiver(latt[id][0], latt[id][1], latt[nn][0]-latt[id][0], latt[nn][1] - latt[id][1])

    ijks = []
    for (id1,nns) in enumerate(nns_list):
        for id2 in nns:
            # if len(nns) < 6:
            #     continue
            nn_p1 = nns_list[id2]
            si = np.intersect1d(nn_p1, nns)
            for (s, id3) in enumerate(si):
                ijks.append([id1, id2, id3])
    Atot = 0
    charge = -999
    for ijk in ijks:
        v1: np.asarray = latt[ijk[1],:] - latt[ijk[0],:]
        v2: np.asarray = latt[ijk[2],:] - latt[ijk[1],:]
        cr = np.cross(v1, v2)
        Oi = np.asarray([Ox.iloc[ijk[0]], Oy.iloc[ijk[0]], Oz.iloc[ijk[0]]])
        Oj = np.asarray([Ox.iloc[ijk[1]], Oy.iloc[ijk[1]], Oz.iloc[ijk[1]]])
        Ok = np.asarray([Ox.iloc[ijk[2]], Oy.iloc[ijk[2]], Oz.iloc[ijk[2]]])
        Xijk = X(Oi, Oj, Ok)
        Yijk = Y(Oi, Oj, Ok)
        Z = (Xijk + 1j*Yijk)
        dA = 2*np.angle(Z)*(np.sign(cr[2]))
        Atot += dA
        charge = Atot/(4*np.pi)/6
    print(charge)