import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

def read_csv_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    qx = data[:, 0].reshape(-1, int(np.sqrt(data.shape[0])))
    qy = data[:, 1].reshape(-1, int(np.sqrt(data.shape[0])))
    S_values = data[:, 2].reshape(-1, int(np.sqrt(data.shape[0])))
    return qx, qy, S_values

def normalize_S_values(S_values):
    # Normalize the S_values to the range [0, 1]
    min_val = np.min(S_values)
    max_val = np.max(S_values)
    return (S_values - min_val) / (max_val - min_val)

def plot_structure_factors(data_dir, output_image):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0, hspace=0)

    cmap = "plasma"
    plot_titles = ["S_{xx}", "S_{xy}", "S_{xz}",
                   "S_{yx}", "S_{yy}", "S_{yz}",
                   "S_{zx}", "S_{zy}", "S_{zz}"]

    for i, title in enumerate(plot_titles):
        csv_filename = os.path.join(data_dir, f"{title}.csv")
        qx, qy, S_values = read_csv_data(csv_filename)

        S_values = normalize_S_values(S_values)
        
        ax = fig.add_subplot(gs[i // 3, i % 3])
        c = ax.pcolor(qx, qy, S_values, shading="auto", cmap=cmap)

        ax.text(np.min(qx), np.max(qy), f"$\\mathbf{{{title}}}$", color="white", fontsize=24, ha="left", va="top")

        ax.set_xticks([])
        ax.set_yticks([])

        if i >= 6:  # Only label the x-axis of the bottom row
            ax.set_xlabel(r"$q_{x} \, a / \pi$",fontsize=20)
        if i % 3 == 0:  # Only label the y-axis of the left column
            ax.set_ylabel(r"$q_{y} \, a / \pi$",fontsize=20)

    fig.subplots_adjust(left=0.11, right=0.89, top=0.89, bottom=0.11)  # Add margin around the grid (opposites must add up to 1)

    # Create a common colorbar that spans the entire width of the figure
    cbar_ax = fig.add_axes([0.11, 0.04, 0.775, 0.03])  
    cbar = fig.colorbar(c, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=18)

    plt.savefig(output_image, dpi=300)
    plt.close(fig)

def calculate_differential_cross_section(data_dir):
    
    Sxx = read_csv_data(os.path.join(data_dir, "S_{xx}.csv"))[2]
    Sxy = read_csv_data(os.path.join(data_dir, "S_{xy}.csv"))[2]
    Sxz = read_csv_data(os.path.join(data_dir, "S_{xz}.csv"))[2]
    Syx = read_csv_data(os.path.join(data_dir, "S_{yx}.csv"))[2]
    Syy = read_csv_data(os.path.join(data_dir, "S_{yy}.csv"))[2]
    Syz = read_csv_data(os.path.join(data_dir, "S_{yz}.csv"))[2]
    Szx = read_csv_data(os.path.join(data_dir, "S_{zx}.csv"))[2]
    Szy = read_csv_data(os.path.join(data_dir, "S_{zy}.csv"))[2]
    Szz = read_csv_data(os.path.join(data_dir, "S_{zz}.csv"))[2]

    qx, qy, _ = read_csv_data(os.path.join(data_dir, "S_{xx}.csv"))
    q_squared = qx**2 + qy**2

    sigma = np.zeros_like(Sxx)

    for i in range(sigma.shape[0]):
        for j in range(sigma.shape[1]):
            qx_val = qx[i, j]
            qy_val = qy[i, j]
            q_squared_val = q_squared[i, j]
            if q_squared_val == 0:
                continue  

            qx2_over_q2 = (qx_val ** 2) / q_squared_val
            qy2_over_q2 = (qy_val ** 2) / q_squared_val
            qxqy_over_q2 = (qx_val * qy_val) / q_squared_val

            sigma[i, j] = (Sxx[i, j] * (1 - qx2_over_q2) + Syy[i, j] * (1 - qy2_over_q2) - 2 * qxqy_over_q2 * Sxy[i, j])

    return qx, qy, sigma

def plot_differential_cross_section(data_dir, output_image):
    qx, qy, sigma = calculate_differential_cross_section(data_dir)

    plt.figure(figsize=(8, 6))
    plt.pcolor(qx, qy, sigma, shading='auto', cmap='plasma')
    plt.colorbar(label=r'$\sigma(q)$')
    plt.xlabel(r"$q_{x} \, a / \pi$")
    plt.ylabel(r"$q_{y} \, a / \pi$")
    plt.title(r"Differential Cross Section $\sigma(\mathbf{q})$")
    plt.savefig(output_image, dpi=300)
    plt.close()

#main
if __name__ == "__main__":
    # if run in REPL from the plots folder
    # data_dir = "../out"  # we need to go one step back otherwise the script will look for the out folder inside plots
    # output_image = "../out/structure_factors.jpg"  
    # plot_structure_factors(data_dir, output_image)
    # output_image = "../out/cross_section.jpg"  

    data_dir = "./out"  
    output_image = "./out/structure_factors.jpg"  

    plot_structure_factors(data_dir, output_image)

    output_image = "./out/cross_section.jpg" 

    plot_differential_cross_section(data_dir, output_image)

    #run from kd_tree_approach folder like: python plots/sans_observables.py
    #print(os.getcwd())    

