import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from auxiliary.hsv2rgb import hsv2rgb

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.latex.preamble"] = r"\usepackage{bm} \usepackage{amsmath}"

def process_dataframe(df, masking=True, max_x=10, max_y=7):
    x, y = df["x"].values, df["y"].values
    s = df["s"].values
    mx, my, mz = df["Mx"].values, df["My"].values, df["Mz"].values

    if masking:
        x_range = (-max_x, max_x)
        y_range = (-max_y, max_y)
        mask = (x >= x_range[0]) & (x <= x_range[1]) & (y >= y_range[0]) & (y <= y_range[1])

        x, y = x[mask], y[mask]
        s = s[mask]
        mx, my, mz = mx[mask], my[mask], mz[mask]

    Lx, Ly = 2 * max_x + 1, 2 * max_y + 1
    X, Y = x.reshape(Lx, Ly), y.reshape(Lx, Ly)
    S_grid = s.reshape(Lx, Ly)
    MX, MY = mx.reshape(Lx, Ly), my.reshape(Lx, Ly)
    
    hue = np.zeros((len(mx), 3))
    for i in range(len(mx)):
        vector = np.array([mx[i], my[i], mz[i]])
        mod = np.linalg.norm(vector)
        hue[i] = hsv2rgb(vector / mod, 1, 0)

    return X, Y, S_grid, MX, MY, hue


def plot_entropies(dfs, w, h):
    grids = [process_dataframe(df) for df in dfs]

    fig, axes = plt.subplots(2, 2, figsize=(w, h))
    axes = axes.flatten()

    subplot_labels = [r"$\rm (a)$", r"$\rm (b)$", r"$\rm (c)$"]
    for i, (ax, (X, Y, S, MX, MY, hue)) in enumerate(zip(axes, grids)):
        ax.set_aspect("equal")
        # cmap = plt.cm.gray_r
        
        if i == 2:
            arw_clr = hue #"whitesmoke"
            cmap = plt.cm.gray_r #"turbo" 
        elif i < 2:
            arw_clr = hue
            cmap = plt.cm.gray_r
            cmap = mcolors.ListedColormap(
                cmap(np.linspace(0.0, 0.65, 256))
            )  # lower bound can be editted for cmaps that go only to gray and not to black
        else:
            arw_clr = hue
            cmap = plt.cm.gray_r
            cmap = mcolors.ListedColormap(
                cmap(np.linspace(0.0, 0.4, 256))
            ) 
        c = ax.pcolormesh(X, Y, S, cmap=cmap, shading="auto")

        ax.quiver(
            X, Y, MX, MY,
            angles="xy",
            scale_units="xy",
            scale=0.4,
            color=arw_clr,
            pivot="middle",
            width=0.007,
        )

        cbar = plt.colorbar(
            c,
            ax=ax,
            orientation="horizontal",
            location="top",
            pad=0.02,
            aspect=40,
        )
        cbar.locator = MaxNLocator(integer=True, prune="both", nbins=3)
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=8)
        
        ax.set_xlabel(r"$x/a$", fontsize=8)
        if i % 2 == 0:
            ax.set_ylabel(r"$y/a$", fontsize=8)
        else:
            ax.set_yticklabels([])
        
        if i < 2:
            cbar.set_label(r"$-\mathrm{Tr}\left(\hat\rho_i\ln\hat\rho_i\right)$", fontsize=8, labelpad=4)
            ax.set_xlabel("")
            ax.set_xticklabels([])

        
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.tick_params(axis="y", which="both", right=True)
        ax.tick_params(axis="both", labelsize=8)

        ax.text(
            0.02, 0.98,
            subplot_labels[i],
            transform=ax.transAxes,
            fontsize=8,
            color="black",
            ha="left",
            va="top",
        )

    plt.tight_layout()
    plt.savefig("entropies.pdf", bbox_inches="tight")


def plot_entropies_mod(dfs, w, h):
    max_x = 9
    max_y = 6
    grids = [process_dataframe(df, masking=True, max_x=max_x, max_y=max_y) for df in dfs[:2]]
    # c is a line plot dataset
    df_line = dfs[2]

    fig = plt.figure(figsize=(w, h))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

    ax0 = fig.add_subplot(gs[0, 0]) 
    ax1 = fig.add_subplot(gs[0, 1])  
    ax2 = fig.add_subplot(gs[1, :])  

    axes = [ax0, ax1, ax2]
    subplot_labels = [r"$\rm (a)$", r"$\rm (b)$", r"$\rm (c)$"]
    symbols = ["^", "v"]


    for i, (ax, (X, Y, S, MX, MY, hue)) in enumerate(zip(axes[:2], grids)):
        ax.set_aspect("equal")
        cmap = plt.cm.gray_r
        
        if i==0:
            cmap = mcolors.ListedColormap(cmap(np.linspace(0.0, 0.5, 256)))
        else:
            cmap = mcolors.ListedColormap(cmap(np.linspace(0.0, 0.9, 256)))
            
        c = ax.pcolormesh(X, Y, S, cmap=cmap, shading="auto")
        ax.quiver(
            X, Y, MX, MY,
            angles="xy",
            scale_units="xy",
            scale=0.5,
            color=hue,
            pivot="middle",
            width=0.01,
        )
        cbar = plt.colorbar(
            c,
            ax=ax,
            orientation="horizontal",
            location="top",
            # pad=0.02,
            aspect=40,
        )
        cbar.locator = MaxNLocator(integer=True, prune="both", nbins=3)
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(r"$-\mathrm{Tr}\left(\hat\rho_i\ln\hat\rho_i\right)$", fontsize=8, labelpad=4)

        ax.set_xlabel(r"$x/a$", fontsize=8)
        if i == 0:
            ax.set_ylabel(r"$y/a$", fontsize=8)
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("")

        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
       
        ax.tick_params(axis="both", labelsize=8)
        ax.text(
            0.03, 0.97,
            subplot_labels[i],
            transform=ax.transAxes,
            fontsize=8,
            color="black",
            ha="left",
            va="top",
        )
        ax.scatter(
            -8, -5,
            marker=symbols[i],
            facecolor="none",
            edgecolor="red",
        )
    
    
    ax2.plot(df_line["amp"]**2, df_line["max_s"], color="blue", lw=1)
    ax2.hlines(np.log(2), -2, 2, colors="gray", linestyles="--", linewidths=1)
    
    ax2.scatter(df_line["amp"][0]**2, df_line["max_s"][0], marker=symbols[0], facecolor='none', edgecolor="red", zorder=10)
    ax2.scatter(1/2, np.log(2)-1e-2, marker=symbols[1], facecolor='none', edgecolor="red", zorder=10)
    
    c = np.linspace(0.01, 0.99, 20)
    S = -c * np.log(c) - (1 - c)*np.log(1 - c)
    ax2.plot(c, S, color="black", linestyle="dotted")
    
    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim([0.0, 1])
    ax2.set_xlabel(r"$c^2$", fontsize=8)
    ax2.set_ylabel(r"$S_{\rm max}$", fontsize=8)
    ax2.tick_params(axis="both", labelsize=8)
    
    # ax2.xaxis.set_major_locator(MaxNLocator(nbins=4))
    # ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))
    

    ax2.text(
        0.01, 0.95,
        subplot_labels[2],
        transform=ax2.transAxes,
        fontsize=8,
        color="black",
        ha="left",
        va="top",
    )
    ax2.text(
        0.85, 0.92,
        r"$\ln(2)$",
        transform=ax2.transAxes,
        fontsize=8,
        color="black",
        ha="left",
        va="top",
    )

    plt.tight_layout()
    plt.savefig("entropies_comb.pdf", bbox_inches="tight")


if __name__ == "__main__":
    
    data_dir = "data"
    # prefacs = ["sk", "ask", "sup", "sym"]
    # dfs = []

    # for pf in prefacs:
    #     df = pd.read_csv(os.path.join(data_dir, "lobs_" + pf + ".csv"))
    #     dfs.append(df)
        
        
    # w = 3 + 3/8  
    # h = w * 0.95
    # plot_entropies(dfs, w, h)
    
    prefacs = ["sk", "sup", "max_s"]
    dfs = []

    for pf in prefacs:
        df = pd.read_csv(os.path.join(data_dir, "lobs_" + pf + ".csv"))
        dfs.append(df)
        
        
    w_tot = 3 + 3/8  
    h_tot = w_tot * 0.85
    plot_entropies_mod(dfs, w_tot, h_tot)


