import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble="\\usepackage{bm,braket}")
# plt.rc("text.latex", preamble='\\usepackage{amsmath}')


def plot_fig2():
    
    data_sk = pd.read_csv("data/sk_states.csv")
    data_ask = pd.read_csv("data/ask_states.csv")

    data_ask_filtered = data_ask[data_ask["E"] >= max(data_sk["E"]) - 0.1]

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{bm,braket,xcolor}")

    w_total = 7  
    h_total = 3.5

    fig = plt.figure(figsize=(w_total, h_total), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1])

    ax0 = fig.add_subplot(gs[:, 0])

    for (M, group), color in zip(data_sk.groupby("M"), colors):
        ax0.plot(group["alpha"], 2 * group["E"], marker="o", markersize=2, label=f"${M}$", color=color)
        ax0.plot(-group["alpha"], 2 * group["E"], marker="o", markersize=2, color=color)

    for (M, group), color in zip(data_ask_filtered.groupby("M"), colors):
        ax0.plot(group["alpha"], 2 * group["E"], marker="o", markersize=2, color=color)
        ax0.plot(-group["alpha"], 2 * group["E"], marker="o", markersize=2, color=color)

    ax0.set_xlabel(r"$\mathrm{DM~anisotropy}, ~ \alpha$")
    ax0.set_ylabel(r"$\mathrm{Energy}, ~ E/J$")
    ax0.set_xlim([-0.25, 0.25])
    ax0.set_ylim([-509.5, -507.2])
    ax0.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax0.grid(True)

    legend = ax0.legend(
        title=r"$\mathrm{Bond~dimension,}~\chi$",
        fontsize=8,
        title_fontsize=8,
        loc="upper center",
    )

    ax0.text(0.02, 0.99, r"\textrm{(a)}", transform=ax0.transAxes, fontsize=9, va="top")

    ax0.text(0.22, 0.35, r"$\mathrm{ASK}$", transform=ax0.transAxes, fontsize=9)
    ax0.text(0.69, 0.35, r"$\mathrm{SK}$", transform=ax0.transAxes, fontsize=9)

    ax0.hlines(
        y=2 * max(data_sk["E"][data_sk["M"] == 16]),
        xmin=-max(data_ask["alpha"]),
        xmax=max(data_ask["alpha"]),
        color="gray",
        linestyle="--",
    )

   
    ax1 = fig.add_subplot(gs[0, 1])

    for M, d in data_sk.groupby("alpha"):
        dsel = d[d["alpha"] < 0.05]
        ax1.plot(1 / dsel["M"], 4 * dsel["sigma"], marker="o", markersize=2)

    ax1.set_ylabel(r"\rm Energy~spread")
    ax1.set_yscale("log")
    ax1.grid(True)
    ax1.text(0.03, 0.95, r"\textrm{(b) SK}", transform=ax1.transAxes, fontsize=9, va="top")

    
    ax2 = fig.add_subplot(gs[1, 1])

    for M, d in data_ask.groupby("alpha"):
        dsel = d[np.abs(d["alpha"] - 0.0) < 0.05]
        if len(dsel) == 0:
            continue
        ax2.plot(
            1 / dsel["M"],
            4 * dsel["sigma"],
            marker="o",
            markersize=2,
            label=f"${np.round(np.unique(dsel['alpha'])[0],2)}$",
        )

    ax2.set_ylabel(r"\rm Energy~spread")
    ax2.set_xlabel(r"\rm Inverse bond dimension, $1/\chi$")
    ax2.set_yscale("log")
    ax2.grid(True)
    ax2.text(0.03, 0.95, r"\textrm{(c) ASK}", transform=ax2.transAxes, fontsize=9, va="top")


    ticks = [0, 1, 2, 4]
    xticks = [1 / 2**i for i in ticks]
    xlabels = [f"$1/{2**i}$" for i in ticks]
    ax1.set_xticks(xticks, labels=xlabels)
    ax2.set_xticks(xticks, labels=xlabels)


    ax2.legend(
        title=r"$\mathrm{DM~anisotropy}~\alpha$",
        fontsize=8,
        title_fontsize=8,
        loc="lower right",
    )

    plt.savefig("cross_spread.pdf")
    plt.close()


def plot_pd(w,h):

    df = pd.read_csv("data/phase_diagram.csv")
    
    font = 10
    msize = 3
    plt.figure(figsize=(w,h))

    plt.plot(
        df["alpha"],
        df["B_sp_FM"],
        color="black",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="blue",
        markersize=msize,
        clip_on=False,
    )

    plt.fill_between(df["alpha"], df["B_sk_DM"], color="red")

    plt.fill_between(
        df["alpha"],
        df["B_sp_FM"],
        color="aquamarine",
    )
    plt.plot(
        df["alpha2"],
        df["B_sk_sp"],
        color="black",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="red",
        markersize=msize,
        clip_on=False,
    )
    plt.fill_between(
        df["alpha2"],
        df["B_sk_sp"],
        color="aquamarine",
    )

    plt.plot(
        df["alpha"],
        df["B_sk_DM"],
        color="black",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="red",
        markersize=msize,
        clip_on=False,
    )

    plt.text(
        0.39,
        0.85,
        r"$\mathrm{FM}$",
        fontsize=font,
        color="black",
        ha="left",
        va="top",
    )

    plt.text(
        0.29,
        0.35,
        r"$\mathrm{Spin-Spiral}$",
        fontsize=font,
        color="black",
        ha="left",
        va="top",
    )

    plt.text(
        0.87,
        0.62,
        r"$\mathrm{SK}$",
        fontsize=font,
        color="black",
        ha="left",
        va="top",
    )

    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.xlabel(r"$\mathrm{DM~anisotropy}, ~ \alpha$", fontsize=font)
    plt.ylabel(r"$\mathrm{Magnetic~field}, ~ B_z$", fontsize=font)
    plt.savefig("phase_diag.jpg", dpi=600, bbox_inches="tight")
    plt.close()


def plot_anisos(w,h):

    J0 = pd.read_csv("data/Results_J0.csv", header=None)
    J2 = pd.read_csv("data/Results_J0m2.csv", header=None)
    J4 = pd.read_csv("data/Results_J0m4.csv", header=None)
    J6 = pd.read_csv("data/Results_J0m6.csv", header=None)
    J8 = pd.read_csv("data/Results_J0m8.csv", header=None)

    font = 10
    plt.figure(figsize=(w,h))
    
    plt.plot(-J0[1], J0[2], label=r"$0$")
    plt.plot(-J2[1], J2[2], label=r"$0.2$")
    plt.plot(-J4[1], J4[2], label=r"$0.4$")
    plt.plot(-J6[1], J6[2], label=r"$0.6$")
    plt.plot(-J8[1], J8[2], label=r"$0.8$")

    plt.xlabel(r"$\mathrm{Magnetic~field}, ~ \vert B_z \vert$", fontsize=font)
    plt.ylabel(r"$\mathrm{Magnetization}, ~ M_z$", fontsize=font)
    plt.legend(title=r"$\tilde J/J$", loc="lower right", fontsize=font, title_fontsize=font)
    plt.grid(True)

    plt.xlim([0, 0.1])
    plt.ylim([0, 1.05])
    plt.savefig("anisos.jpg", dpi=600, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    w = 2.4
    h = 2.4
    # plot_pd(w,h)
    # plot_anisos(w,h)
    
    plot_fig2()
    