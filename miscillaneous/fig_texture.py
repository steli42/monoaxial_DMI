import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble="\\usepackage{bm,braket}")
# plt.rc("text.latex", preamble='\\usepackage{amsmath}')


def plot_texture():
    
    df = pd.read_csv("data/phase_diagram.csv")
    J0 = pd.read_csv("data/Results_J0.csv", header=None)
    J2 = pd.read_csv("data/Results_J0m2.csv", header=None)
    J4 = pd.read_csv("data/Results_J0m4.csv", header=None)
    J6 = pd.read_csv("data/Results_J0m6.csv", header=None)
    J8 = pd.read_csv("data/Results_J0m8.csv", header=None)

    font = 8
    msize = 3
    w_total = 5.2
    h_total = 2.5
    D = 2 * np.pi / 15.0
    B_D = D**2/ 2.0 
    
    fig, axs = plt.subplots(1, 2, figsize=(w_total, h_total))
    
    axs[0].plot(
        df["alpha"],
        df["B_sp_FM"],
        color="black",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=msize,
        clip_on=False,
    )
    
    axs[0].fill_between(df["alpha"], df["B_sk_DM"], color="red")

    axs[0].fill_between(
        df["alpha"],
        df["B_sp_FM"],
        color="aquamarine",
    )
    axs[0].plot(
        df["alpha2"],
        df["B_sk_sp"],
        color="black",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=msize,
        clip_on=False,
    )
    axs[0].fill_between(
        df["alpha2"],
        df["B_sk_sp"],
        color="aquamarine",
    )

    axs[0].plot(
        df["alpha"],
        df["B_sk_DM"],
        color="black",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=msize,
        clip_on=False,
    )

    axs[0].text(
        0.39,
        0.85,
        r"$\mathrm{FM}$",
        fontsize=font+2,
        color="black",
        ha="left",
        va="top",
    )

    axs[0].text(
        0.29,
        0.35,
        r"$\mathrm{Spin-Spiral}$",
        fontsize=font+2,
        color="black",
        ha="left",
        va="top",
    )

    axs[0].text(
        0.87,
        0.62,
        r"$\mathrm{SL}$",
        fontsize=font+2,
        color="black",
        ha="left",
        va="top",
    )
    
    axs[0].scatter(
        [0.7],
        [0.6175],
        marker="s",
        facecolor="gold",
        edgecolor="black",
        s=20,
        zorder=10)

    axs[0].set_ylim([0, 1])
    axs[0].set_xlim([0, 1])
    axs[0].set_xlabel(r"$\mathrm{DM~anisotropy}, ~ \alpha$", fontsize=font)
    axs[0].set_ylabel(r"$\mathrm{Magnetic~field}, ~ B_z/B_D$", fontsize=font)
    
    # second panel 
    
    axs[1].plot(-J0[1]/B_D, J0[2], label=r"$0$")
    axs[1].plot(-J2[1]/B_D, J2[2], label=r"$0.2$")
    axs[1].plot(-J4[1]/B_D, J4[2], label=r"$0.4$")
    axs[1].plot(-J6[1]/B_D, J6[2], label=r"$0.6$")
    axs[1].plot(-J8[1]/B_D, J8[2], label=r"$0.8$")

    axs[1].set_xlabel(r"$\mathrm{Magnetic~field}, ~ \vert B_z/B_D \vert$", fontsize=font)
    axs[1].set_ylabel(r"$\mathrm{Magnetization}, ~ M_z$", fontsize=font)
    axs[1].legend(title=r"$\tilde J/J$", loc="lower right", fontsize=font, title_fontsize=font)
    axs[1].grid(True, linestyle="--")

    axs[1].set_xlim([0, 1.0])
    axs[1].set_ylim([0, 1.05])
    for ax in axs:
        ax.tick_params(axis="both", labelsize=font)

    
    plt.tight_layout()
    plt.savefig("fig1.pdf", bbox_inches="tight")
    plt.close()
    
    
if __name__ == "__main__":

    
    plot_texture()
    
    