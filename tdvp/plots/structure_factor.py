import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fn = "neel/corr.csv"
df = pd.read_csv(fn)

ks = np.linspace(-np.pi, np.pi, 20)
qhats = np.asarray([[kx, ky, 0]/np.linalg.norm([kx, ky, 0]) for kx in ks for ky in ks])
Sab = {}
j=0


obs = ["Sx", "Sy", "Sz"]
data = df
x, y = np.meshgrid(ks, ks)
for (lbl,p) in zip(['im','re'],[np.imag, np.real]):
    fig, axs = plt.subplots(ncols=len(obs),nrows=len(obs),sharex=True,sharey=True)
    axs = axs.ravel()
    i=0
    for o1 in obs:
        for o2 in obs:
            data[f'{o1}*{o2}'] = data[f'{o1}*{o2}_re'] + 1j*data[f'{o1}*{o2}_im']
            S = np.array([sum(data[f'{o1}*{o2}']*np.exp(1j*kx*(data["x'"]-data["x"]))*np.exp(1j*ky*(data["y'"]-data["y"]))) for kx in ks for ky in ks])
            Sab[o1,o2] = S
            im = axs[i].imshow(np.reshape(p(S), (len(ks), len(ks))), cmap='viridis', interpolation='quadric', origin='lower', extent=[-1,1,-1,1])
            axs[i].text(0.08, 0.9, "$S_{"+f"{o1}{o2}".replace("S","")+"}$", transform=axs[i].transAxes, fontsize=6)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            i+=1
    fig.subplots_adjust(left=None, bottom=None, right=1.0, top=1.0, wspace=-0.36, hspace=0.0)
    cbar = fig.colorbar(im, extend='both', location='bottom', ax=axs.tolist(), pad=0.02, shrink=0.825)

    fn_repl = fn.replace('.csv',f'_{lbl}.jpg')
    plt.savefig(fn_repl, bbox_inches='tight', dpi=600, pad_inches=0)
    # plt.show()
    plt.close()

## perhaps useful -- scattering cross section
# dsdO = 0*Sab[obs[0],obs[0]]
# for (a,o1) in enumerate(obs[0:3]):
#     for (b,o2) in enumerate(obs[0:3]):
#         dab = 0.0
#         if o1==o2: dab=1.0
#         dsdO += [(dab - qhat[a]*qhat[b])*Sab[o1,o2][idx] for (idx, qhat) in enumerate(qhats)]
# im = axs_cs[j].imshow(np.reshape(np.real(dsdO), (len(ks), len(ks))), cmap='viridis', interpolation='quadric', origin='lower', extent=[-1,1,-1,1])
# axs_cs[j].text(0.02, 0.85, f"${labels[j]}$", transform=axs_cs[j].transAxes, fontsize=10, c='white')
# theta = np.linspace(-np.pi, np.pi, 100)
# if j==2:
#     for r in [0.304917, 0.558283]:
#         x = r * np.sin(theta)
#         y = r * np.cos(theta)
#         axs_cs[j].plot(x, y, color='white', linewidth=0.5, linestyle='dotted')
# j+=1
# axs_cs[0].set_xlabel("$q_x/\pi$")
# axs_cs[1].set_xlabel("$q_x/\pi$")
# axs_cs[2].set_xlabel("$q_x/\pi$")
# axs_cs[0].set_ylabel("$q_y/\pi$")
# fig_cs.colorbar(im, ax=axs_cs.tolist(), location='bottom', extend='both', pad=0.175, aspect=40)
# plt.savefig(f'plots/cross_section.jpg', bbox_inches='tight', dpi=600, pad_inches=0)
# # plt.show()
# plt.close()