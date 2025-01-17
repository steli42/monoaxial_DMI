import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fns = ["sk16/state_corr.csv", "sk16/conj_state_corr.csv", "sk16/ss1_corr.csv", "sk16/ss2_corr.csv"]
for fn in fns:
    df = pd.read_csv(fn)
    data = df

    ks = np.linspace(-1, 1, 100)
    qhats = np.asarray([[kx, ky, 0]/np.linalg.norm([kx, ky, 0]) for kx in ks for ky in ks])
    Sab = {}
    Sabconn = {}
    j=0

    kxs = np.array([kx for kx in ks for ky in ks])
    kys = np.array([ky for kx in ks for ky in ks])


    obs = ["Sx", "Sy", "Sz"]
    x, y = np.meshgrid(ks, ks)
    dfq = pd.DataFrame()
    dfq["q_x"] = kxs
    dfq["q_y"] = kys
    for o1 in obs:
        for o2 in obs:
            print(o1, o2)
            data[f'{o1}*{o2}'] = (data[f'{o1}*{o2}_re'] + 1j*data[f'{o1}*{o2}_im'])
            data[f'{o1}*{o2}_conn'] = data[f'{o1}*{o2}'] - data[f'{o1}*Id_re']*data[f'Id*{o2}_re']
            Sconn = np.array([sum(data[f'{o1}*{o2}_conn']*np.exp(1j*kx*(data["x'"]-data["x"]))*np.exp(1j*ky*(data["y'"]-data["y"]))) for kx in ks for ky in ks])
            Sabconn[o1,o2] = Sconn
            S = np.array([sum(data[f'{o1}*{o2}']*np.exp(1j*kx*(data["x'"]-data["x"]))*np.exp(1j*ky*(data["y'"]-data["y"]))) for kx in ks for ky in ks])
            Sab[o1,o2] = S

    for (lbl,p) in zip(['im','re'],[np.imag, np.real]):
        smin = 0
        smax = 0
        im = []
        for o1 in obs:
            for o2 in obs:
                # if lbl == 'im':
                #     continue
                dfq[f"{lbl}({o1} {o2})"] = p(Sab[o1,o2])
                dfq[f"{lbl}({o1} {o2})_c"] = p(Sabconn[o1,o2])
                smin = min(smin, min(p(Sab[o1,o2])))
                smax = max(smax, max(p(Sab[o1,o2])))
                print(o1,o2,lbl,min(p(Sab[o1,o2])),max(p(Sab[o1,o2])), sep='\t')
    fn_repl = fn.replace('corr',f'sfac')
    dfq.to_csv(fn_repl)