#!/usr/bin/env python
# coding: utf-8
import numpy as np

from auxi import find_files as ff
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib as mpl

plt.rc('text', usetex=True)
plt.rc('text.latex',preamble='\\usepackage{bm,braket}')

mpl.rcParams['figure.figsize'] = (4, 4)

in_dir = '.'

str = "/energy"
fns = np.sort(ff.find_files(f'*{str}.csv', in_dir))

fig, ax = plt.subplots(1,1)

for (idfn, fn) in enumerate(fns):
    data = pd.read_csv(fn)
    print(data["E_sk"])
    ax.scatter(data["sigma_sk"].iloc[0], data["E_sk"].iloc[0]+123.1526, color='black', marker='x')

ax.set_xlabel('$\\braket{\\hat H - E}^2$')
ax.set_ylabel('$(E-E_\\infty)/|J|$')
ax.set_xscale('log')
ax.set_yscale('log')
plt.tight_layout()
plt.savefig('energy_convergence.pdf')