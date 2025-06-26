import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble="\\usepackage{bm,braket}")
# plt.rc("text.latex", preamble='\\usepackage{amsmath}')


def plot_energies():
    data_all1 = pd.read_csv("sk_states.csv")
    data_all2 = pd.read_csv("ask_states.csv")

    data_all2 = data_all2[data_all2["E"] >= max(data_all1["E"]) - 0.1]

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.hlines(
        y=2 * max(data_all1["E"][data_all1["M"] == 16]),
        xmin=-max(data_all2["alpha"]),
        xmax=max(data_all2["alpha"]),
        color="gray",
        linestyle="--",
    )
    for (M, group), color in zip(data_all1.groupby("M"), colors):
        plt.plot(
            group["alpha"],
            2.0 * group["E"],
            label=f"${M}$",
            marker="o",
            markersize=4,
            color=color,
        )
        plt.plot(
            -group["alpha"],
            2.0 * group["E"],
            marker="o",
            markersize=4,
            color=color,
        )
    plt.legend()
    for (M, group), color in zip(data_all2.groupby("M"), colors):
        plt.plot(
            group["alpha"],
            2.0 * group["E"],
            marker="o",
            markersize=4,
            color=color,
        )
        plt.plot(
            -group["alpha"],
            2.0 * group["E"],
            marker="o",
            markersize=4,
            color=color,
        )

    plt.xlabel(r"$\mathrm{DMI~anisotropy}, ~ \alpha$", fontsize=16)
    plt.ylabel(r"$\mathrm{Energy}, ~ E / J$", fontsize=16)

    plt.grid(True)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    plt.tick_params(labelsize=16)
    legend = ax.legend(
        title=r"$\mathrm{Bond~dimension}$",
        fontsize=16,
        title_fontsize=16,
        loc="upper right",
    )

    # ax.text(
    #     0.3,
    #     0.98,
    #     r"$\mathrm{Antiskyrmion}$",
    #     transform=ax.transAxes,
    #     fontsize=16,
    #     color="black",
    #     ha="left",
    #     va="top",
    # )
    
    ax.text(
        0.22,
        0.98,
        r"$\mathrm{Antiskyrmion}$",
        transform=ax.transAxes,
        fontsize=16,
        color="black",
        ha="left",
        va="top",
    )

    # ax.text(
    #     0.29,
    #     0.55,
    #     r"$\mathrm{Skyrmion}$",
    #     transform=ax.transAxes,
    #     fontsize=16,
    #     color="black",
    #     ha="left",
    #     va="top",
    # )
    
    ax.text(
        0.24,
        0.68,
        r"$\mathrm{Skyrmion}$",
        transform=ax.transAxes,
        fontsize=16,
        color="black",
        ha="left",
        va="top",
    )


    plt.xlim([-0.25,0.25])
    plt.ylim([-510,-507])
    plt.setp(legend.get_texts(), fontsize=16)  # Adjusts legend item fontsize
    plt.setp(legend.get_title(), fontsize=16)  # Title font size and bold

    plt.savefig("energy.png", dpi=600, bbox_inches="tight")
    plt.close()


def plot_pd():

    df = pd.read_csv("phase_diagram.csv")
    plt.figure()

    plt.plot(
        df["alpha"],
        df["B_sp_FM"],
        color="black",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="blue",
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
        clip_on=False,
    )

    plt.text(
        0.39,
        0.85,
        r"$\mathrm{FM}$",
        fontsize=30,
        color="black",
        ha="left",
        va="top",
    )

    plt.text(
        0.29,
        0.35,
        r"$\mathrm{Spin-spiral}$",
        fontsize=30,
        color="black",
        ha="left",
        va="top",
    )

    plt.text(
        0.87,
        0.62,
        r"$\mathrm{SK}$",
        fontsize=30,
        color="black",
        ha="left",
        va="top",
    )

    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.xlabel(r"$\mathrm{DMI~anisotropy}, ~ \alpha$", fontsize=16)
    plt.ylabel(r"$\mathrm{Magnetic~field}, ~ B_z$", fontsize=16)
    plt.savefig("phase_diag.png", dpi=600, bbox_inches="tight")
    plt.close()


def plot_anisos():

    J0 = pd.read_csv("Results_J0.csv", header=None)
    J2 = pd.read_csv("Results_J0m2.csv", header=None)
    J4 = pd.read_csv("Results_J0m4.csv", header=None)
    J6 = pd.read_csv("Results_J0m6.csv", header=None)
    J8 = pd.read_csv("Results_J0m8.csv", header=None)

    plt.figure()
    plt.plot(-J0[1], J0[2], label=r"$0$")
    plt.plot(-J2[1], J2[2], label=r"$0.2$")
    plt.plot(-J4[1], J4[2], label=r"$0.4$")
    plt.plot(-J6[1], J6[2], label=r"$0.6$")
    plt.plot(-J8[1], J8[2], label=r"$0.8$")
    
    plt.xlabel(r"$\mathrm{Magnetic~field}, ~ \vert B_z \vert$", fontsize=16)
    plt.ylabel(r"$\mathrm{Magnetization}, ~ M_z$", fontsize=16)
    plt.legend(title = r"$\tilde J/J$", loc="lower right", fontsize=16, title_fontsize=16)
    plt.grid(True)
    
    plt.xlim([0,0.1])
    plt.ylim([0,1.05])
    plt.savefig("anisos.png", dpi=600, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    plot_energies()
    # plot_pd()
    # plot_anisos()
