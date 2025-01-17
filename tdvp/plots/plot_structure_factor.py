import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from auxi import export_legend as el
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rc('text', usetex=True)
plt.rc('text.latex',preamble='\\usepackage{bm,xcolor}')

fns = ["sk16/state_sfac.csv", "sk16/ss1_sfac.csv", "sk16/ss2_sfac.csv", "sk16/conj_state_sfac.csv"]
dfs = [pd.read_csv(fn) for fn in fns]


ks = np.unique(dfs[0]["q_x"])

qhats = np.asarray([[kx, ky, 0]/np.linalg.norm([kx, ky, 0]) for kx in ks for ky in ks])
Sab = {}
Sabconn = {}
j=0

cmap = 'inferno'

obs = ["Sx", "Sy", "Sz"]
for (fn,df) in zip(fns, dfs):
    ks = np.unique(df["q_x"])
    qhats = np.asarray([[kx, ky, 0]/np.linalg.norm([kx, ky, 0]) for kx in ks for ky in ks])
    for (lbl,p) in zip(['im','re'],[np.imag, np.real]):
        # continue
        fig, axs = plt.subplots(ncols=len(obs),nrows=len(obs),sharex=True,sharey=True)
        axs = axs.ravel()
        i=0
        smin = 0
        smax = 0
        im = []
        for o1 in obs:
            for o2 in obs:
                vals = np.sign(df[f"{lbl}({o1} {o2})"])*np.log(np.abs(df[f"{lbl}({o1} {o2})"]))
                vals = df[f"{lbl}({o1} {o2})"]

                smin = min(smin, min(vals))
                smax = max(smax, max(vals))
                print(o1,o2,lbl,min(vals),max(vals), sep='\t')

                im.append(axs[i].imshow(np.reshape(vals, (len(ks), len(ks))).T, cmap=cmap, interpolation='quadric', origin='lower', extent=[-1,1,-1,1], vmin=-300, vmax=1000))
                axs[i].text(0.08, 0.9, "$\\mathcal S_{"+f"{o1}{o2}".replace("S","")+"}$", transform=axs[i].transAxes, fontsize=6, color='white')
                axs[i].set_xticks([])
                axs[i].set_yticks([])
                i+=1
        imgs = [ax.get_images() for ax in axs]
        fig.subplots_adjust(left=None, bottom=None, right=1.0, top=1.0, wspace=-0.483, hspace=0.0)

        fn_repl = fn.replace('.csv',f'_{lbl}.jpg')
        plt.savefig(fn_repl, bbox_inches='tight', dpi=600, pad_inches=0)
        # plt.show()
        plt.close()

fns = ["sk16/state_sfac.csv", "sk16/ss1_sfac.csv", "sk16/conj_state_sfac.csv"]
dfs = [pd.read_csv(fn) for fn in fns]
dpi = 600
w = 3.25
h = 3.25/3
fig, axs = plt.subplots(1, 3, figsize=(w, h), sharey=True)
axs = axs.ravel()
i = 0
for (fn,df) in zip(fns, dfs):
    ks = np.unique(df["q_x"])
    qhats = np.asarray([[kx, ky, 0]/np.linalg.norm([kx, ky, 0]) for kx in ks for ky in ks])

    ax = axs[i]
    dsdO = 0*df["re(Sx Sx)"]
    for (a,o1) in enumerate(obs[:2]):
        for (b,o2) in enumerate(obs[:2]):
            sfac = (df[f"re({o1} {o2})"] + 1j*df[f"im({o1} {o2})"])
            dab = 0.0
            if o1==o2: dab=1.0
            dsdO += [(dab - qhat[a]*qhat[b])*sfac[idx] for (idx, qhat) in enumerate(qhats)]
    vals = np.log(np.real(dsdO))
    print(min(vals),max(vals))
    imag = ax.imshow(np.reshape(vals, (len(ks), len(ks))).T, cmap=cmap, interpolation='quadric', origin='lower', extent=[min(ks),max(ks),min(ks),max(ks)], vmin=4.6, vmax=7)
    ax.set_xlabel("$q_x\,a$")
    # fig.colorbar(im, ax=ax, location='bottom', extend='both', pad=0.175, aspect=40)
    # plt.tight_layout()

    # axins = inset_axes(
    #     ax,
    #     width="100%",  # width: 5% of parent_bbox width
    #     height="5%",  # height: 50%
    #     loc="lower left",
    #     bbox_to_anchor=(0., 1.0, 1, 1),
    #     bbox_transform=ax.transAxes,
    #     borderpad=0
    # )
    # cbar = fig.colorbar(imag, cax=axins, orientation='horizontal', ticklocation='top')
    # cbar.ax.set_title('$\\ln(\\sigma_p)$')
    i += 1
axs[0].text(-0.9, -0.9, "$c=0$", color='white')
axs[1].text(-0.9, -0.9, "$c=1/\\sqrt2$", color='white')
axs[2].text(-0.9, -0.9, "$c=1$", color='white')
axs[0].set_ylabel("$q_y\,a$")
# plt.tight_layout()
plt.savefig("polarized_sans.pdf", bbox_inches='tight', dpi=600, pad_inches=0)
plt.savefig("polarized_sans.jpeg", bbox_inches='tight', dpi=600, pad_inches=0)
# plt.show()
plt.close()

fig, axs = plt.subplots(1, 3, figsize=(w, h), sharey=True)
axs = axs.ravel()
i = 0
for (fn,df) in zip(fns, dfs):
    ks = np.unique(df["q_x"])
    qhats = np.asarray([[kx, ky, 0]/np.linalg.norm([kx, ky, 0]) for kx in ks for ky in ks])

    ax = axs[i]
    dsdO = 0*df["re(Sx Sx)"]
    for (a,o1) in enumerate(obs):
        for (b,o2) in enumerate(obs):
            sfac = (df[f"re({o1} {o2})_c"] + 1j*df[f"im({o1} {o2})_c"])
            dab = 0.0
            if o1==o2: dab=1.0
            dsdO += [(dab - qhat[a]*qhat[b])*sfac[idx] for (idx, qhat) in enumerate(qhats)]
    print(fn)
    vals = np.log(np.real(dsdO))
    print(min(vals),max(vals))
    imag = ax.imshow(np.reshape(vals, (len(ks), len(ks))).T, cmap=cmap, interpolation='quadric', origin='lower', extent=[min(ks),max(ks),min(ks),max(ks)], vmin = 4.6, vmax = 6.2)
    ax.set_xlabel("$q_x\,a$")
    # fig.colorbar(im, ax=ax, location='bottom', extend='both', pad=0.175, aspect=40)
    # plt.tight_layout()

    # axins = inset_axes(
    #     ax,
    #     width="100%",  # width: 5% of parent_bbox width
    #     height="5%",  # height: 50%
    #     loc="lower left",
    #     bbox_to_anchor=(0., 1.0, 1, 1),
    #     bbox_transform=ax.transAxes,
    #     borderpad=0
    # )
    # cbar = fig.colorbar(imag, cax=axins, orientation='horizontal', ticklocation='top')
    # cbar.ax.set_title('$\\ln(\\sigma_p)$')
    i += 1
axs[0].text(-0.9, -0.9, "$c=0$", color='white')
axs[1].text(-0.9, -0.9, "$c=1/\\sqrt2$", color='white')
axs[2].text(-0.9, -0.9, "$c=1$", color='white')
axs[0].set_ylabel("$q_y\,a$")
# plt.tight_layout()
plt.savefig("polarized_sans_connected.pdf", bbox_inches='tight', dpi=600, pad_inches=0)
plt.savefig("polarized_sans_connected.jpeg", bbox_inches='tight', dpi=600, pad_inches=0)
# plt.show()
plt.close()