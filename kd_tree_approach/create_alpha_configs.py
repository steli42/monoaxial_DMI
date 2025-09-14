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

# create a fresh configs directory where the json files will be stored
dir = "configs"
dir_out = "/mnt/lscratch/users/sliscak"
if not os.path.isdir(dir):
    os.makedirs(dir)
else:
    shutil.rmtree(dir)
    os.makedirs(dir)

# look for all files /globs.csv in the alphas folder    
in_dir = '/mnt/lscratch/users/sliscak/alphas'   
file = '/globs'
# store the sorted filenames (paths) to all globs files 
fns = np.sort(find_files(f'*{file}.csv', in_dir))

# concatinate the contents of all globs files into df_all
# in addition add the paths under the tag 'fn'
df_all = pd.DataFrame()
for fn in fns:
    df = pd.read_csv(fn)
    df['fn'] = fn
    df_all = pd.concat([df_all, df], ignore_index=True)

f = open('default.json')
data = json.load(f)

# if we want to use pre-saved MPS files, it must be fixed to 50
# otherwise the searched for fn strings do not match and the finder errors out
alphas = np.linspace(0, 1, 50) 
Ms = [16]
Bs = [1.2 + i*2e-3 for i in range(0,6)]

# ctr = 0
# for alpha in alphas:
#     for bonddim in Ms:
#         # generate a unique folder name (unique identifier) 
#         # this way even the same M and alpha do not clash 
#         uuid_str = uuid.uuid1()
        
#         # inject new values in the default json
#         data['output_dir'] = f"{dir_out}/asks/"+str(uuid_str) 
#         data['alpha'] = alpha
#         data['bonddim'] = bonddim
        
        
#         # filter df_all such that the M column only holds values equal to 'bonddim'
#         df_sel = df_all[df_all['M'] == bonddim]
#         # same for alpha but with a twist since alpha is a float
#         df_sel = df_sel[abs(df_sel['alpha'] - alpha) < 1e-3]
        
#         # using the filtered dataframe, strip away the /globs.csv string from the globs path
#         # basically just leaves the directory path of the corresponding (alpha, M) folder
#         initial_mps_dir = df_sel['fn'].iloc[0].replace("globs.csv", "")
#         # in the given folder find the .h5 file 
#         mps_fn = np.sort(find_files(f'*.h5', initial_mps_dir))[0]
        
#         # inject filename of initial mps 
#         data['mps_initial_fn'] = mps_fn

#         with open(f'{dir}/cfg_'+f'{ctr}'.zfill(5)+'.json', 'w') as file:
#                     json.dump(data, file, ensure_ascii=False, indent=4)

#         ctr += 1
        
        
ctr = 0
for alpha in alphas:
    for bonddim in Ms:
        for B in Bs:
            uuid_str = uuid.uuid1()
            data['output_dir'] = f"{dir_out}/fields/"+str(uuid_str) 
            data['alpha'] = alpha
            data['bonddim'] = bonddim
            data['B_amp'] = B
            
            df_sel = df_all[df_all['M'] == bonddim]
            df_sel = df_sel[abs(df_sel['alpha'] - alpha) < 1e-3]
            initial_mps_dir = df_sel['fn'].iloc[0].replace("globs.csv", "")
            mps_fn = np.sort(find_files(f'*.h5', initial_mps_dir))[0]
            data['mps_initial_fn'] = mps_fn

            with open(f'{dir}/cfg_'+f'{ctr}'.zfill(5)+'.json', 'w') as file:
                        json.dump(data, file, ensure_ascii=False, indent=4)

            ctr += 1
