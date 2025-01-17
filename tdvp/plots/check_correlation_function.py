import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fn = "sk16/state_corr.csv"
df = pd.read_csv(fn)
data1 = df

fn = "sk16/conj_state_corr.csv"
df = pd.read_csv(fn)
data2 = df

fn = "sk16/mixed_corr.csv"
df = pd.read_csv(fn)
data3 = df

fn = "sk16/symmetric_superposition_corr.csv"
df = pd.read_csv(fn)
data_symm = df

c = 1.0/np.sqrt(2)
phi = np.pi/3
obs = ["Sx", "Sy", "Sz"]
for o1 in obs:
    for o2 in obs:
        data1[f'{o1}*{o2}'] = (data1[f'{o1}*{o2}_re'] + 1j*data1[f'{o1}*{o2}_im'])
        data2[f'{o1}*{o2}'] = (data2[f'{o1}*{o2}_re'] + 1j*data2[f'{o1}*{o2}_im'])
        data3[f'{o1}*{o2}'] = (data3[f'{o1}*{o2}_re'] + 1j*data3[f'{o1}*{o2}_im'])
        data_symm[f'{o1}*{o2}'] = (data_symm[f'{o1}*{o2}_re'] + 1j*data_symm[f'{o1}*{o2}_im'])

        mixed = 0*c*np.sqrt(1-c**2)*np.exp(1j*phi)*data3[f'{o1}*{o2}']
        tp = c**2*data1[f'{o1}*{o2}'] + (1-c**2)*data2[f'{o1}*{o2}'] + mixed + np.conj(mixed)
        print(data_symm[f'{o1}*{o2}'] - tp)