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

    subplot_labels = [r"$\rm (a)$", r"$\rm (b)$", r"$\rm (c)$", r"$\rm (d)$"]
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


if __name__ == "__main__":
    
    data_dir = "data"
    prefacs = ["sk", "ask", "sup", "sym"]
    dfs = []

    for pf in prefacs:
        df = pd.read_csv(os.path.join(data_dir, "lobs_" + pf + ".csv"))
        dfs.append(df)
        
        
    w = 3 + 3/8  
    h = w * 0.95
    plot_entropies(dfs, w, h)


