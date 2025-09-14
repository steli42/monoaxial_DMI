import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.latex.preamble"] = r"\usepackage{bm} \usepackage{amsmath}"


def color_fun(H):
    result = 0
    if 0 <= H < 60:
        result = H / 60
    elif 60 <= H < 180:
        result = 1
    elif 240 <= H <= 360:
        result = 0
    elif 180 <= H < 240:
        result = 4 - H / 60
    return result


def hsv2rgb(n, in_v, in_h):
    nom = np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2) + np.finfo(float).eps
    F = np.arctan2(n[1] / nom, n[0] / nom)
    H = 360 * in_h + (1 - 2 * in_h) * (F if F >= 0 else 2 * np.pi + F) * 180 / np.pi
    H = H % 360

    m1 = 1 - abs(n[2]) / nom if (1 - 2 * in_v) * n[2] / nom < 0 else 1
    m2 = 0 if (1 - 2 * in_v) * n[2] / nom < 0 else abs(n[2]) / nom

    max_v = 0.5 + nom * (m1 - 0.5)
    min_v = 0.5 - nom * (0.5 - m2)
    dV = max_v - min_v

    rgb = list(n)
    rgb[0] = np.round(color_fun((H + 120) % 360) * dV + min_v, decimals=10)
    rgb[1] = np.round(color_fun(H % 360) * dV + min_v, decimals=10)
    rgb[2] = np.round(color_fun((H - 120) % 360) * dV + min_v, decimals=10)

    return rgb


def plot_entropy_fidelity(X, Y, MX, MY, s_grid, Delta_grid, norm_grid, hue, save_path):

    # gr = (1+np.sqrt(5))/2.0
    # dpi = 600
    w = 3*(3+3/8)*0.5
    h = (3+3/8)*0.5
    fig, axes = plt.subplots(1, 3, figsize=(w, h))
    # fig, axes = plt.subplots(1, 3, constrained_layout=True)
    data_grids = [s_grid, Delta_grid, norm_grid]
    labels = [
        r"$-\mathrm{Tr}\left(\hat\rho_i\ln\hat\rho_i\right)$",
        r"$\mathrm{Tr}(\hat\rho_i \hat P_i)$",
        r"$\vert \bm{m}_i \vert$",
    ]
    subplot_labels = [r"$\rm (d)$", r"$\rm (k)$", r"$\rm (l)$"]
    label_colors = ["black", "white", "black"]

    for i, (ax, data, label) in enumerate(zip(axes, data_grids, labels)):

        if i < 2:
            arw_clr = hue #"whitesmoke"
            cmap = plt.cm.gray_r #"turbo" 
        else:
            arw_clr = hue
            cmap = plt.cm.grey
            cmap = mcolors.ListedColormap(
                cmap(np.linspace(0.0, 1, 256))
            )  # lower bound can be editted for cmaps that go only to gray and not to black

        ax.set_aspect("equal")
        c = ax.pcolormesh(X, Y, data, cmap=cmap, shading="auto")
        ax.quiver(
            X,
            Y,
            MX,
            MY,
            angles="xy",
            scale_units="xy",
            scale=0.5,
            color=arw_clr,
            pivot="middle",
            width=0.0045,
        )
        cbar = plt.colorbar(
            c,
            ax=ax,
            orientation="horizontal",
            location="top",
            pad=0.005,
            aspect=45,
            shrink=1.0,
        )
        # cbar.set_label(label, fontsize=10)
        cbar.locator = MaxNLocator(integer=True, prune="both", nbins=3)
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=10)
        ax.set_xlabel(r"$x/a$", fontsize = 10)
        # ax.set_xticklabels([])
        if i == 0:
            ax.set_ylabel(r"$y/a$")
        elif i != 0:
            ax.set_yticklabels([])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis="y", which="both", right=True)

        ax.text(
            0.02,
            0.98,
            subplot_labels[i],
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            color=label_colors[i],
            ha="left",
            va="top",
        )
    plt.savefig(save_path, dpi=600, bbox_inches="tight")


prefac = "sup"
data_dir = os.path.join("..", "data_lobs")
df = pd.read_csv(os.path.join(data_dir, "lobs_" + prefac + ".csv"))

x, y = df["x"].values, df["y"].values
mx, my, mz = df["Mx"].values, df["My"].values, df["Mz"].values
s, Delta = df["s"].values, df["Delta"].values
norm = np.sqrt(mx**2 + my**2 + mz**2)

masking = True
max_x = 10
max_y = 7

if masking != True:
    Lx, Ly = len(np.unique(x)), len(np.unique(y))
    X, Y = x.reshape(Lx, Ly), y.reshape(Lx, Ly)
    MX, MY, MZ = mx.reshape(Lx, Ly), my.reshape(Lx, Ly), mz.reshape(Lx, Ly)
    s_grid, Delta_grid = s.reshape(Lx, Ly), Delta.reshape(Lx, Ly)
    norm_grid = norm.reshape(Lx, Ly)
elif masking == True:

    x_range = (-max_x, max_x)
    y_range = (-max_y, max_y)

    mask = (x >= x_range[0]) & (x <= x_range[1]) & (y >= y_range[0]) & (y <= y_range[1])

    x = x[mask]
    y = y[mask]
    mx = mx[mask]
    my = my[mask]
    mz = mz[mask]
    s = s[mask]
    Delta = Delta[mask]
    norm = norm[mask]

    Lx, Ly = 2 * max_x + 1, 2 * max_y + 1
    X, Y = x.reshape(Lx, Ly), y.reshape(Lx, Ly)
    MX, MY, MZ = mx.reshape(Lx, Ly), my.reshape(Lx, Ly), mz.reshape(Lx, Ly)
    s_grid, Delta_grid = s.reshape(Lx, Ly), Delta.reshape(Lx, Ly)
    norm_grid = norm.reshape(Lx, Ly)

    hue = np.zeros((len(mx), 3))
    for i in range(len(mx)):
        vector = np.array([mx[i], my[i], mz[i]])
        mod = np.linalg.norm(vector)
        hue[i] = hsv2rgb(vector / mod, 1, 0)

# Generate the plot
plot_entropy_fidelity(
    X,
    Y,
    MX,
    MY,
    s_grid,
    Delta_grid,
    norm_grid,
    hue,
    os.path.join(data_dir, "combined_plot_" + prefac + ".pdf"),
)
