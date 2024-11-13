import json
import numpy as np
import os, shutil
import uuid

def xyz(phi, r):
    return np.asfarray([r*np.cos(phi), r*np.sin(phi), 0])

# import matplotlib.pyplot as plt

dir = "configs"
dir_out = "/work/projects/tmqs_projects"
if not os.path.isdir(dir):
    os.makedirs(dir)
else:
    shutil.rmtree(dir)
    os.makedirs(dir)

f = open('cfg/default.json')
data = json.load(f)

alphas = np.linspace(0,1,48)
Ms = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]


ctr = 0
for alpha in alphas:
    for bonddim in Ms:
        uuid_str = uuid.uuid1()
        data['io_dir'] = f"{dir_out}/alphas/"+str(uuid_str)
        data['alpha'] = alpha
        data['M'] = bonddim

        with open(f'{dir}/cfg_'+f'{ctr}'.zfill(5)+'.json', 'w') as file:
                    json.dump(data, file, ensure_ascii=False, indent=4)

        ctr += 1
