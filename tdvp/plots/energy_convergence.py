#!/usr/bin/env python
# coding: utf-8
import numpy as np

from auxi import find_files as ff
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib as mpl

import json

# plt.rc('text', usetex=True)
# plt.rc('text.latex',preamble='\\usepackage{bm,braket}')

mpl.rcParams['figure.figsize'] = (4, 4)

in_dir = 'alphas'

str = "/energy"
fns = np.sort(ff.find_files(f'*{str}.csv', in_dir))

fig, ax = plt.subplots(1,1)

data_all = []
for (idfn, fn) in enumerate(fns):
    data = pd.read_csv(fn)
    fn_repl = fn.replace('energy.csv', 'params.json')
    p = json.load(open(fn_repl))
    data['alpha'] = p['alpha']
    data['M'] = p['M']

    # if(p['M']==40): print(fn)
    data_all.append(data)
data_all = pd.concat(data_all)
data_all.to_csv('all_energies.csv')
# exit()

Ms = np.unique(data_all['M'])[-1:]
print(Ms)
for bonddim in Ms:
    data_sel = data_all[data_all['M'] == bonddim]
    ax.scatter(data_sel['alpha'], abs(data_sel['<sk|ask>_re']+1j*data_sel['<sk|ask>_im']))

# ax.set_xlabel('$\\braket{\\hat H - E}^2$')
# ax.set_ylabel('$(E-E_\\infty)/|J|$')
# ax.set_xscale('log')
ax.set_yscale('log')
plt.tight_layout()
plt.savefig('energy_convergence.pdf')