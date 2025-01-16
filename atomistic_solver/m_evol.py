import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm  

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

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.axis('off')  # Turn off axis labels and ticks

    cmap = plt.cm.RdBu_r

    # map mz values to interval [0, 1]
    mz_normalized = (m[:,2] + 1) / 2
    z_color = mz_normalized.reshape(Lx, Ly)

    X,Y = lattice[:,0].reshape(Lx, Ly),lattice[:,1].reshape(Lx, Ly)
    ax.pcolormesh(X, Y, z_color, cmap=cmap)

    ax.quiver(X, Y, m[:,0].reshape(Lx, Ly), m[:,1].reshape(Lx, Ly), 
                angles='xy', scale_units='xy', scale=1, color='white', pivot='middle', width=0.0025)


    plt.tight_layout()
    plt.savefig(os.path.join(snapshot_dir, f"shot_{snapshot_id}.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__": 
    
    file = "atomistic_solver/m_evol_flattened.h5"

    with h5py.File(file, "r") as h5file:
        lattice = h5file["lattice"][:]
        m_evol = h5file["m_evol"][:].T
        P = int(h5file["P"][()])
        Lx = int(h5file["Lx"][()])
        Ly = int(h5file["Ly"][()])

    # Ensure the snapshots directory exists
    snapshot_dir = os.path.join("atomistic_solver", "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)

    # Clear all files in the directory
    for filename in os.listdir(snapshot_dir):
        file_path = os.path.join(snapshot_dir, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'w') as file:
                pass  # Clear the file content
            


    # Iterate through the snapshots and plot
    incr = 100
    id = 1 
    for i in tqdm(range(0, P, incr), desc="Generating snapshots"):
        plot_magnetization_2D(snapshot_dir, lattice, m_evol[i], id, Lx, Ly)
        id += 1



# ffmpeg -framerate 24 -i snapshots/shot_%d.png -c:v libx264 -pix_fmt yuv420p movie.mp4


