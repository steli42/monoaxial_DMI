import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


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


data_dir = os.path.join("..", "data_lobs")
df = pd.read_csv(os.path.join(data_dir, "lobs.csv"))

x, y = df["x"].values, df["y"].values
mx, my, mz = df["Mx"].values, df["My"].values, df["Mz"].values
s, Delta = df["s"].values, df["Delta"].values

Lx, Ly = len(np.unique(x)), len(np.unique(y))
X, Y = x.reshape(Lx, Ly), y.reshape(Lx, Ly)
MX, MY, MZ = mx.reshape(Lx, Ly), my.reshape(Lx, Ly), mz.reshape(Lx, Ly)
s_grid, Delta_grid = s.reshape(Lx, Ly), Delta.reshape(Lx, Ly)

hue = np.zeros((len(mx), 3))
for i in range(len(mx)):
    vector = np.array([mx[i], my[i], mz[i]])
    norm = np.linalg.norm(vector)
    hue[i] = hsv2rgb(vector / norm, 1, 0)

mz_normalized = (mz + 1) / 2
z_color = mz_normalized.reshape(Lx, Ly)

norm_code = np.sqrt(mx**2 + my**2 + mz**2)
norm_code = norm_code.reshape(Lx, Ly)

fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.pcolormesh(X, Y, norm_code, cmap="gray")  # ocean #bone #gray work

ax.quiver(
    X,
    Y,
    MX,
    MY,
    angles="xy",
    scale_units="xy",
    scale=0.5,
    color=hue,
    pivot="middle",
    width=0.0045,
)

ax.set_xlabel("x")
ax.set_ylabel("y")
plt.savefig(os.path.join(data_dir, "magnetization.png"), dpi=600)

fig, ax = plt.subplots()
ax.set_aspect("equal")
c = ax.pcolormesh(X, Y, s_grid, cmap="turbo", shading="auto")
ax.quiver(
    X,
    Y,
    MX,
    MY,
    angles="xy",
    scale_units="xy",
    scale=0.5,
    color="white",
    pivot="middle",
    width=0.0025,
)
# plt.colorbar(c, ax = ax, label="s")
cbar = plt.colorbar(
    c,
    ax=ax,
    label="entropy",
    orientation="horizontal",
    location="top",
    pad=0.01,
    aspect=75,
)
fig.tight_layout()
plt.savefig(os.path.join(data_dir, "entropy.png"), dpi=600)

fig, ax = plt.subplots()
ax.set_aspect("equal")
c = ax.pcolormesh(X, Y, Delta_grid, cmap="plasma_r", shading="auto")
ax.quiver(
    X,
    Y,
    MX,
    MY,
    angles="xy",
    scale_units="xy",
    scale=0.5,
    color="white",
    pivot="middle",
    width=0.0045,
)
cbar = plt.colorbar(
    c,
    ax=ax,
    label="Delta",
    orientation="horizontal",
    location="top",
    pad=0.01,
    aspect=75,
)
fig.tight_layout()
plt.savefig(os.path.join(data_dir, "fidelity.png"), dpi=600)
