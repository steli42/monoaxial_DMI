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
    h_total = 2.8

    fig = plt.figure(figsize=(w_total, h_total), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.5, 1], height_ratios=[1, 1])

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
    ax0.set_ylim([-509.5, -507])
    ax0.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax0.grid(True)

    legend = ax0.legend(
        title=r"$\mathrm{Bond~dimension,}~\chi$",
        fontsize=7,
        title_fontsize=7,
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
    # ax1.set_yscale("log")
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
    # ax2.set_yscale("log")
    ax2.grid(True)
    ax2.text(0.03, 0.95, r"\textrm{(c) ASK}", transform=ax2.transAxes, fontsize=9, va="top")


    ticks = [0, 1, 2, 4]
    xticks = [1 / 2**i for i in ticks]
    xlabels = [f"$1/{2**i}$" for i in ticks]
    ax1.set_xticks(xticks, labels=xlabels)
    ax2.set_xticks(xticks, labels=xlabels)


    ax2.legend(
        title=r"$\mathrm{DM~anisotropy}~\alpha$",
        fontsize=7,
        title_fontsize=7,
        loc="lower right",
    )

    plt.savefig("cross_spread.pdf")
    plt.close()

if __name__ == "__main__":

    plot_fig2()
    