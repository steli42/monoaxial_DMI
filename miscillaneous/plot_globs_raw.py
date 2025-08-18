#!/usr/bin/env python
# coding: utf-8
import os, sys
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from icecream import ic

# plt.rc('text', usetex=True)
# plt.rc('text.latex',preamble='\\usepackage{bm,braket}')

fn = 'data/ask_states.csv'
data_all = pd.read_csv(fn)

fig, ax = plt.subplots(1,1)


df = pd.DataFrame(columns=['inv_M', 'alpha_crit'])
for M,group in data_all.groupby("M"):
    slice = group[group["alpha"]<0.35]
    slice2 = slice[slice['E'] == max(slice['E'])]
    
    inv_M = 1./slice2["M"].values[0]
    alpha_crit = slice2["alpha"].values[0]
    
    df.loc[len(df)] = [inv_M, alpha_crit]
plt.scatter(df['inv_M'], df['alpha_crit'])

coeffs, cov = np.polyfit(df['inv_M'], df['alpha_crit'], deg=1, cov=True)
intercept_err = np.sqrt(cov[1, 1])

x = np.linspace(-0.1,1)
plt.plot(x, coeffs[0] * x + coeffs[1], '--k')
plt.hlines( coeffs[1], -0.1, 1.0, linestyle="--", color='red')
plt.hlines( coeffs[1]-intercept_err, -0.1, 1.0, linestyle="dotted", color='red')
plt.hlines( coeffs[1]+intercept_err, -0.1, 1.0, linestyle="dotted", color='red')
plt.grid(True)
plt.xlim([0.0,1.05])
plt.ylim([0.0,0.3])
plt.savefig("trend.png",dpi=600)
plt.close()   
    

ic(coeffs[1])
ic(intercept_err)





