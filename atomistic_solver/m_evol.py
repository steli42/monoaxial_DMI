import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm  # for the progress bars
import numpy as np
import os
import glob

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



def plot_magnetization_2D(snapshot_dir, lattice, m, snapshot_id, Lx, Ly):
    """
    Function to plot a 2D magnetization snapshot.

    Parameters:
        snapshot_dir (str): Directory to save the snapshots.
        lattice (numpy.ndarray): 2D array representing the lattice positions.
        m (numpy.ndarray): Array of magnetization vectors.
        snapshot_id (int): Identifier for the snapshot.
        Lx and Ly (int): Lattice dimensions.
    """
    m = m.reshape((-1, 3))

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.set_aspect("equal")
    ax.axis("off")  # Turn off axis labels and ticks

    cmap = plt.cm.RdBu_r

    # map mz values to interval [0, 1]
    mz_normalized = (m[:, 2] + 1) / 2
    z_color = mz_normalized.reshape(Lx, Ly)
    
    hue = np.zeros((len(m[:,0]), 3))
    for i in range(len(m[:,0])):
        vector = np.array([m[i,0], m[i,1], m[i,2]])
        mod = np.linalg.norm(vector)
        hue[i] = hsv2rgb(vector / mod, 1, 0) 

    X, Y = lattice[:, 0].reshape(Lx, Ly), lattice[:, 1].reshape(Lx, Ly)
    ax.pcolormesh(X, Y, z_color, cmap=cmap)

    ax.quiver(
        X,
        Y,
        m[:, 0].reshape(Lx, Ly),
        m[:, 1].reshape(Lx, Ly),
        angles="xy",
        scale_units="xy",
        scale=1,
        color="black",
        pivot="middle",
        width=0.005,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(snapshot_dir, f"shot_{snapshot_id}.png"), dpi=600)
    plt.close(fig)


if __name__ == "__main__":

    file = "./m_evol.h5"
    # file = "./m_relaxation.h5"

    with h5py.File(file, "r") as h5file:
        lattice = h5file["lattice"][:]
        m_evol = h5file["m_evol"][:].T
        P = int(h5file["P"][()])
        Lx = int(h5file["Lx"][()])
        Ly = int(h5file["Ly"][()])

    snapshot_dir = os.path.join(".", "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)

    for file in glob.glob(os.path.join(snapshot_dir, "*.png")):
        os.remove(file)

    incr = 100
    id = 1
    for i in tqdm(range(0, P, incr), desc="Generating snapshots"):
        plot_magnetization_2D(snapshot_dir, lattice, m_evol[i], id, Lx, Ly)
        id += 1


# ffmpeg -framerate 64 -i snapshots/shot_%d.png -c:v libx264 -pix_fmt yuv420p movie.mp4
