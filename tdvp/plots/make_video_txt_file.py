import json
import numpy as np
import os, shutil
import uuid

from auxi import find_files as ff

folder = 'sk'
string = '*series/*mag.png'
fns = ff.find_files(f'{string}', folder)

fns = np.sort(fns)
alphas = []
for fn in fns:
    fn_rpl = fn.replace("mag.png", "params.json")
    # params = json.load(open(fn_rpl))
    # alphas.append(params['alpha'])
# alphas = np.asarray(alphas)

# seld = (abs(alphas) < 0.02)
# print(fns[seld])

# sel = (alphas > 0.0)
# alphas = alphas[sel]
# fns = fns[sel]
# order = np.flip(np.argsort(alphas))
# alphas = alphas[order]
# fns = fns[order]
# print(alphas)
with open('video.txt', 'w') as f:
    for fn in fns:
        f.write(f"file {fn}\n")