import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from auxi import export_legend as el
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rc('text', usetex=True)
plt.rc('text.latex',preamble='\\usepackage{bm}')

fn = "sk16/sfac.csv"
df = pd.read_csv(fn)


ks = np.unique(df["q_x"])

qhats = np.asarray([[kx, ky, 0]/np.linalg.norm([kx, ky, 0]) for kx in ks for ky in ks])
Sab = {}
Sabconn = {}
j=0

cmap = 'viridis'

obs = ["Sx", "Sy", "Sz"]

for (lbl,p) in zip(['im','re'],[np.imag, np.real]):
    fig, axs = plt.subplots(ncols=len(obs),nrows=len(obs),sharex=True,sharey=True)
    axs = axs.ravel()
    i=0
    smin = 0
    smax = 0
    im = []
    for o1 in obs:
        for o2 in obs:
            # if lbl == 'im':
            #     continue
            sign = +1
            # if o1=="Sy": sign *= -1
            # if o2=="Sy": sign *= -1
            vals = np.sign(df[f"{lbl}({o1} {o2})"])*np.log(np.abs(df[f"{lbl}({o1} {o2})"]))
            vals = sign*df[f"{lbl}({o1} {o2})"]

            smin = min(smin, min(vals))
            smax = max(smax, max(vals))
            print(o1,o2,lbl,min(vals),max(vals), sep='\t')

            im.append(axs[i].imshow(np.reshape(vals, (len(ks), len(ks))).T, cmap=cmap, interpolation='quadric', origin='lower', extent=[-1,1,-1,1]))
            axs[i].text(0.08, 0.9, "$S_{"+f"{o1}{o2}".replace("S","")+"}$", transform=axs[i].transAxes, fontsize=6)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            i+=1
    imgs = [ax.get_images() for ax in axs]
    # print(smin, smax)
    # print([i[0].get_clim() for i in imgs])
    # [i[0].set_clim([smin, smax]) for i in imgs]
    # cbar = fig.colorbar(im[len(im)-1], extend='both', location='bottom', ax=axs.tolist(), pad=0.02, shrink=0.825)
    # plt.tight_layout()
    fig.subplots_adjust(left=None, bottom=None, right=1.0, top=1.0, wspace=-0.36, hspace=0.0)

    fn_repl = fn.replace('.csv',f'_{lbl}.jpg')
    plt.savefig(fn_repl, bbox_inches='tight', dpi=600, pad_inches=0)
    # plt.show()
    plt.close()

# perhaps useful -- scattering cross section
gr = (1+np.sqrt(5))/2.0
dpi = 600
w = 3.25/2
h = 3.25/2
fig, ax = plt.subplots(figsize=(w, h))

dsdO = 0*df["re(Sx Sx)"]
for (a,o1) in enumerate(obs[0:2]):
    for (b,o2) in enumerate(obs[0:2]):
        sign = +1
        # if o1=="Sy": sign *= -1
        # if o2=="Sy": sign *= -1
        Sabconn = sign*(df[f"re({o1} {o2})"] + 1j*df[f"im({o1} {o2})"])
        dab = 0.0
        if o1==o2: dab=1.0
        dsdO += [(dab - qhat[a]*qhat[b])*Sabconn[idx] for (idx, qhat) in enumerate(qhats)]
print(min(np.imag(dsdO)),max(np.imag(dsdO)))
print(min(np.real(dsdO)),max(np.real(dsdO)))
vals = np.log(np.real(dsdO))
imag = ax.imshow(np.reshape(vals, (len(ks), len(ks))).T, cmap=cmap, interpolation='quadric', origin='lower', extent=[min(ks),max(ks),min(ks),max(ks)])
ax.set_xlabel("$q_x\,a$")
ax.set_ylabel("$q_y\,a$")
# fig.colorbar(im, ax=ax, location='bottom', extend='both', pad=0.175, aspect=40)
fn_repl = fn.replace('.csv',f'_scs.jpg')
# plt.tight_layout()

axins = inset_axes(
    ax,
    width="100%",  # width: 5% of parent_bbox width
    height="5%",  # height: 50%
    loc="lower left",
    bbox_to_anchor=(0., 1.0, 1, 1),
    bbox_transform=ax.transAxes,
    borderpad=0
)
# cmap = mpl.cm.viridis
# norm = mpl.colors.Normalize(vmin=0, vmax=1)
cbar = fig.colorbar(imag, cax=axins, orientation='horizontal', ticklocation='top')
# cbar.ax.set_title('$\\ln(\\sigma_p)$')

# plt.tight_layout()
plt.savefig(fn_repl, dpi=600, pad_inches=0)
plt.savefig(fn_repl, bbox_inches='tight', dpi=600, pad_inches=0)
# plt.show()
plt.close()