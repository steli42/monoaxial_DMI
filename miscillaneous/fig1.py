import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble="\\usepackage{bm,braket}")
# plt.rc("text.latex", preamble='\\usepackage{amsmath}')


def plot_energies(w,h):
    data_all1 = pd.read_csv("data/sk_states.csv")
    data_all2 = pd.read_csv("data/ask_states.csv")

    data_all2 = data_all2[data_all2["E"] >= max(data_all1["E"]) - 0.1]

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    font = 10
    markersize = 2
    fig, ax = plt.subplots(figsize=(w, h))

    for (M, group), color in zip(data_all1.groupby("M"), colors):
        plt.plot(
            group["alpha"],
            2.0 * group["E"],
            label=f"${M}$",
            marker="o",
            markersize=markersize,
            color=color,
        )
        plt.plot(
            -group["alpha"],
            2.0 * group["E"],
            marker="o",
            markersize=markersize,
            color=color,
        )
    plt.legend()
    for (M, group), color in zip(data_all2.groupby("M"), colors):
        plt.plot(
            group["alpha"],
            2.0 * group["E"],
            marker="o",
            markersize=markersize,
            color=color,
        )
        plt.plot(
            -group["alpha"],
            2.0 * group["E"],
            marker="o",
            markersize=markersize,
            color=color,
        )

    plt.xlabel(r"$\mathrm{DM~anisotropy}, ~ \alpha$", fontsize=font)
    plt.ylabel(r"$\mathrm{Energy}, ~ E / J$", fontsize=font)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    plt.tick_params(labelsize=10)
    legend = ax.legend(
        title=r"$\mathrm{Bond~dimension,}~\chi$",
        fontsize=font,
        title_fontsize=font,
        loc="lower center",
    )
    
    ax.text(
        0.02,
        0.99,
        "\\rm (a)",
        transform=ax.transAxes,
        fontsize=font,
        color="black",
        ha="left",
        va="top",
    )

    ax.text(
        0.22,
        0.65,
        r"$\mathrm{Ask}$",
        transform=ax.transAxes,
        fontsize=font,
        color="black",
        ha="left",
        va="top",
    )

    ax.text(
        0.69,
        0.65,
        r"$\mathrm{Sk}$",
        transform=ax.transAxes,
        fontsize=font,
        color="black",
        ha="left",
        va="top",
    )

    plt.hlines(
        y=2 * max(data_all1["E"][data_all1["M"] == 16]),
        xmin=-max(data_all2["alpha"]),
        xmax=max(data_all2["alpha"]),
        color="gray",
        linestyle="--",
    )

    plt.xlim([-0.3, 0.3])
    plt.ylim([-510.9, -507.4])
    plt.setp(legend.get_texts(), fontsize=font)  # Adjusts legend item fontsize
    plt.setp(legend.get_title(), fontsize=font)  # Title font size and bold
    plt.grid(True)
    plt.savefig("energy.pdf", bbox_inches="tight")
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

    w = 3 + 3 / 8
    h = 3 + 3 / 8
    plot_energies(w,h)
    
    # w = 2.4
    # h = 2.4
    # plot_pd(w,h)
    # plot_anisos(w,h)
