#!/usr/bin/env python
# coding: utf-8
import numpy as np
from auxiliary import find_files as ff
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import json

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble="\\usepackage{bm,braket,xcolor}")

fn1 = "elipse_01.csv"
fn2 = "elipse_05.csv"
fn3 = "elipse_025.csv"
fn4 = "elipse_abr.csv"

d1 = pd.read_csv(fn1, header=None)
d2 = pd.read_csv(fn2, header=None)
d3 = pd.read_csv(fn3, header=None)
d4 = pd.read_csv(fn4, header=None)

gr = (1 + np.sqrt(5)) / 2.0
font = 10
w = gr * (3 + 3 / 8) * 0.82
h = (3 + 3 / 8) * 0.82
fig, ax = plt.subplots(1, 1, figsize=(w, h))

ax.plot(d4[0], d4[1], label="$\\rm sudden\\ quench$")
ax.plot(d3[0], d3[1], label="$0.05$")
ax.plot(d2[0], d2[1], label="$0.025$")
ax.plot(d1[0], d1[1], label="$0.01$")

# print(d1[3])
ax.plot(d1[0], d1[3] - 1, color="black", linestyle="dotted")
ax.set_xlim(0, 1000)
ax.set_ylim(-1, 9)

# axs[0].set_ylabel('\\rm energy\\ spread')
# axs[0].set_xlabel('$1/\\chi$')
# axs[0].text(0.15,0.36,'$\\rm skyrmion$')
# axs[1].text(0.15,0.36,'$\\rm antiskyrmion$')
# axs[0].text(0.05,0.36,'$\\rm (a)$')
# axs[1].text(0.05,0.36,'$\\rm (b)$')
# ticks = [0, 1, 2, 4]
# axs[0].set_xticks([1/2**i for i in ticks], labels=[f'$1/{2**i}$' for i in ticks])
# axs[1].set_xticks([1/2**i for i in ticks], labels=[f'$1/{2**i}$' for i in ticks])
ax.set_ylabel("$\\rm skyrmion\\ center$ $x_{\\rm sk}/a$", fontsize=font)
ax.set_xlabel("$\\rm dimensionless\\ time$ $t|J|/s$", fontsize=font)
plt.tight_layout()
# plt.legend(title='${\\rm temporal\\ modulation\\ }p$', loc='lower right')
plt.legend(title="${\\rm temporal\\ modulation\\ }p$", loc="upper left", fontsize=font)
plt.savefig("x_com_t_2.jpg", dpi=600, bbox_inches="tight")
