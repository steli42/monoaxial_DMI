#!/usr/bin/env python
# coding: utf-8
import os, sys
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib as mpl

def find_files(find_str, pwd_path):
    """
        detect files in folder pwd_path that match find_str
    """
    message = "searching for " + find_str + ": "
    sys.stdout.write(message)
    filenames = []
    for filename in os.popen("find " + str(pwd_path) + " -path "
                            + '"' + str(find_str) + '"').read().split('\n')[0:-1]:
        filenames.append(filename)
    message = str(len(filenames)) + " files found.\n"
    sys.stdout.write(message)
    return filenames


# plt.rc('text', usetex=True)
# plt.rc('text.latex',preamble='\\usepackage{bm,braket}')

in_dir = '/mnt/lscratch/users/sliscak/alphas'

file = '/globs'
fns = np.sort(find_files(f'*{file}.csv', in_dir))

fig, ax = plt.subplots(1,1)

data_all = []
for (idfn, fn) in enumerate(fns):
    data = pd.read_csv(fn)
    data_all.append(data)
    
data_all = pd.concat(data_all)
data_all.to_csv(os.path.join(in_dir,"all_energies.csv"))


plt.figure()
for alpha,group in data_all.groupby("alpha"):
    plt.plot(1./group["M"],group["sigma"],label=f"alpha = {alpha}", marker='o')
plt.xlabel("1/M")
plt.ylabel("sigma^2")
plt.grid(True)
plt.savefig("spread.png",dpi=600)

plt.figure()
for M,group in data_all.groupby("M"):
    # plt.plot(group["alpha"],group["Ec"],label=f"M = {M}", marker='x')
    plt.plot(group["alpha"],group["E"],label=f"M = {M}", marker='o')
plt.xlabel("alpha")
plt.ylabel("E")
plt.grid(True)
plt.savefig("trendK1.png",dpi=600)




# Ms = np.unique(data_all['M'])[-1:]
# print(Ms)
# for bonddim in Ms:
#     data_sel = data_all[data_all['M'] == bonddim]
#     ax.scatter(data_sel['alpha'], abs(data_sel['<sk|ask>_re']+1j*data_sel['<sk|ask>_im']))

# ax.set_xlabel('$\\braket{\\hat H - E}^2$')
# ax.set_ylabel('$(E-E_\\infty)/|J|$')
# ax.set_xscale('log')
# ax.set_yscale('log')
# plt.tight_layout()
# plt.savefig('energy_convergence.pdf')