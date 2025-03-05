import json
import numpy as np
import os, shutil
import uuid
import sys
import pandas as pd

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

dir = "configs"
dir_out = "/mnt/lscratch/users/sliscak"
if not os.path.isdir(dir):
    os.makedirs(dir)
else:
    shutil.rmtree(dir)
    os.makedirs(dir)
    
in_dir = '/mnt/lscratch/users/sliscak/alphas'   
file = '/globs'
fns = np.sort(find_files(f'*{file}.csv', in_dir))

df_all = pd.DataFrame()
for fn in fns:
    df = pd.read_csv(fn)
    df['fn'] = fn
    df_all = pd.concat([df_all, df], ignore_index=True)

f = open('default.json')
data = json.load(f)

alphas = np.linspace(0, 1, 50) #now it must be fixed to 50, otherwise the sought for fn strings do not match and the finder errors out
Ms = [1,2,4,8]


ctr = 0
for alpha in alphas:
    for bonddim in Ms:
        uuid_str = uuid.uuid1()
        data['output_dir'] = f"{dir_out}/asks/"+str(uuid_str) # /asks is a new dir housing results of dmrgx targetting ask states
        data['alpha'] = alpha
        data['bonddim'] = bonddim
        
        df_sel = df_all[df_all['M'] == bonddim]
        df_sel = df_sel[abs(df_sel['alpha'] - alpha) < 1e-3]
        initial_mps_dir = df_sel['fn'].iloc[0].replace("globs.csv", "")
        mps_fn = np.sort(find_files(f'*.h5', initial_mps_dir))[0]
        
        data['mps_initial_fn'] = mps_fn

        with open(f'{dir}/cfg_'+f'{ctr}'.zfill(5)+'.json', 'w') as file:
                    json.dump(data, file, ensure_ascii=False, indent=4)

        ctr += 1
