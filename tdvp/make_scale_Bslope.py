import json
import numpy as np
import os, shutil
import uuid

def xyz(phi, r):
    return np.asfarray([r*np.cos(phi), r*np.sin(phi), 0])

# import matplotlib.pyplot as plt

dir = "tdvp/configs"
dir_out = "."
if not os.path.isdir(dir):
    os.makedirs(dir)
else:
    shutil.rmtree(dir)
    os.makedirs(dir)

f = open('tdvp/cfg/default.json')
data = json.load(f)

slopes = np.linspace(0.0001,0.1,32)


ctr = 0
for slope in slopes:
    uuid_str = uuid.uuid1()
    data['io_dir'] = f"{dir_out}/scale_slope/"+str(uuid_str)
    data['Bgrad_slope'] = slope

    # uuid_str = ctr
    # data['io_dir'] = f"braid_{data['boundary_conditions']}_{data['lattice']}/"+str(uuid_str).zfill(4)

    # if ctr > 0:
    #      data['initial_MPS'] = 'MPS'
    #      data['hdf5_initial'] = f"braid_{data['boundary_conditions']}_{data['lattice']}/"+str(uuid_str-1).zfill(4)+"/state.h5"

    with open(f'{dir}/cfg_'+f'{ctr}'.zfill(5)+'.json', 'w') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)

    ctr += 1
