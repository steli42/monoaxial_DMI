import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

data_dir = os.path.join("..", "data_lobs")
df = pd.read_csv(os.path.join(data_dir, "lobs.csv"))

x, y = df["x"].values, df["y"].values
Mx, My, Mz = df["Mx"].values, df["My"].values, df["Mz"].values
s, Delta = df["s"].values, df["Delta"].values

Lx = len(np.unique(x))
Ly = len(np.unique(y))
X, Y = x.reshape(Lx, Ly), y.reshape(Lx, Ly)
s_grid = s.reshape(Lx, Ly)
Delta_grid = Delta.reshape(Lx, Ly)


fig, ax = plt.subplots()
ax.set_aspect("equal")
cmap = plt.cm.RdBu_r
mz_normalized = (Mz + 1) / 2
z_color = mz_normalized.reshape(Lx, Ly)

ax.pcolormesh(X, Y, z_color, cmap=cmap)

ax.quiver(
    X,
    Y,
    Mx.reshape(Lx, Ly),
    My.reshape(Lx, Ly),
    angles="xy",
    scale_units="xy",
    scale=0.5,
    color="white",
    pivot="middle",
    width=0.0025,
)

ax.set_xlabel("x")
ax.set_ylabel("y")
plt.savefig(os.path.join(data_dir, "magnetization.png"), dpi = 600)

fig, ax = plt.subplots()
ax.set_aspect("equal")
c = ax.pcolormesh(X, Y, s_grid, cmap="turbo", shading="auto")
ax.quiver(
    X,
    Y,
    Mx.reshape(Lx, Ly),
    My.reshape(Lx, Ly),
    angles="xy",
    scale_units="xy",
    scale=0.5,
    color="white",
    pivot="middle",
    width=0.0025,
)
# plt.colorbar(c, ax = ax, label="s")
cbar = plt.colorbar(c, ax=ax, label="entropy", orientation = "horizontal", location = "top", pad=0.01, aspect = 75)
fig.tight_layout()
plt.savefig(os.path.join(data_dir, "entropy.png"), dpi = 600)

fig, ax = plt.subplots()
ax.set_aspect("equal")
c = ax.pcolormesh(X, Y, Delta_grid, cmap="turbo", shading="auto")
ax.quiver(
    X,
    Y,
    Mx.reshape(Lx, Ly),
    My.reshape(Lx, Ly),
    angles="xy",
    scale_units="xy",
    scale=0.5,
    color="white",
    pivot="middle",
    width=0.0025,
)
cbar = plt.colorbar(c, ax=ax, label="Delta", orientation = "horizontal", location = "top", pad=0.01, aspect = 75)
fig.tight_layout()
plt.savefig(os.path.join(data_dir, "fidelity.png"), dpi = 600)


