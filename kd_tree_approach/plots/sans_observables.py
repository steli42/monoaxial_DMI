import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

def read_csv_data(file_path, data_type):
    data = np.loadtxt(file_path, delimiter=',')
    num_points = data.shape[0]
    n = int(np.sqrt(num_points))
    if n * n != num_points:
        raise ValueError("Data points are not in a square configuration")   #This might be reduntant; we always take a square of q-values
    
    qx = data[:, 0].reshape(-1, int(np.sqrt(data.shape[0])))
    qy = data[:, 1].reshape(-1, int(np.sqrt(data.shape[0])))
    
    if data_type == 'Re':
        S_values = data[:, 2].reshape(-1, int(np.sqrt(data.shape[0])))
    elif data_type == 'Im':
        S_values = data[:, 3].reshape(-1, int(np.sqrt(data.shape[0])))
    elif data_type == 'Norm':
        S_values = data[:, 4].reshape(-1, int(np.sqrt(data.shape[0])))
    else:
        raise ValueError("Invalid data_type. Choose from 'Re', 'Im' or 'Norm'.")
        
    return qx, qy, S_values

def normalize_S_values(S_values, norm_const):
    # Normalize the S_values to the range [0, 1]
    min_val = np.min(S_values)
    max_val = np.max(S_values)  
    
    if max_val == min_val:
        if max_val == 0.0:
            return np.full_like(S_values, 0.0)
        return np.full_like(S_values, norm_const)    
    return (S_values - min_val) / (max_val - min_val)

def plot_structure_factors(data_dir, output_image, norm_const, data_type='Re', cmap="plasma", vmin=None, vmax=None, log_scale=False):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0, hspace=0)

    cmap = cmap
    plot_titles = ["S_{xx}", "S_{xy}", "S_{xz}",
                   "S_{yx}", "S_{yy}", "S_{yz}",
                   "S_{zx}", "S_{zy}", "S_{zz}"]

    for i, title in enumerate(plot_titles):
        csv_filename = os.path.join(data_dir, f"{title}.csv")
        qx, qy, S_values = read_csv_data(csv_filename, data_type)

        if data_type in ['Re','Im']:
            S_values = normalize_S_values(S_values, norm_const)
        
        if data_type == 'Norm':
            if log_scale:
                S_values = np.where(S_values > 0, S_values, np.nan)  # Replace non-positive values with NaN
                S_values = np.log10(S_values)
                cbar_label = r'$\log|S|$'        
            else:
                cbar_label = r'$|S|$'
        else:
            if log_scale:
                # Note: For 'Re' and 'Im', log scale is not typically used
                S_values = np.where(S_values > 0, S_values, np.nan)  
                S_values = np.log10(S_values)
                cbar_label = r'$\log($' + data_type + '$(S))$' 
            else:
                cbar_label = data_type + r'${}(S)$'         
        
        ax = fig.add_subplot(gs[i // 3, i % 3])
        c = ax.pcolor(qx, qy, S_values, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)

        ax.text(np.min(qx) + 0.1, np.max(qy) - 0.1, f"${title}$", color="white", fontsize=24, ha="left", va="top")

        ax.set_xticks([])
        ax.set_yticks([])

        if i >= 6:  # Only label the x-axis of the bottom row
            ax.set_xlabel(r"$q_{x} \, a / \pi$",fontsize=20)
        if i % 3 == 0:  # Only label the y-axis of the left column
            ax.set_ylabel(r"$q_{y} \, a / \pi$",fontsize=20)

    fig.subplots_adjust(left=0.11, right=0.89, top=0.89, bottom=0.11)  # Add margin around the grid (opposites must add up to 1)

    # Create a common colorbar that spans the entire width of the figure
    cbar_ax = fig.add_axes([0.11, 0.04, 0.775, 0.03])  
    cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal', extend='max')
    cbar.ax.tick_params(labelsize=18)
    plt.text(0.1, 0.04, cbar_label, rotation=0, ha='right', va='bottom', fontsize=20, transform=fig.transFigure)

    plt.savefig(output_image, dpi=600)
    plt.close(fig)

def calculate_differential_cross_section(data_dir):
    Sxx = read_csv_data(os.path.join(data_dir, "S_{xx}.csv"), 'Re')[2]
    Sxy = read_csv_data(os.path.join(data_dir, "S_{xy}.csv"), 'Re')[2]
    Syx = read_csv_data(os.path.join(data_dir, "S_{yx}.csv"), 'Re')[2]
    Syy = read_csv_data(os.path.join(data_dir, "S_{yy}.csv"), 'Re')[2]
    Szz = read_csv_data(os.path.join(data_dir, "S_{zz}.csv"), 'Re')[2]

    qx, qy, _ = read_csv_data(os.path.join(data_dir, "S_{xx}.csv"), 'Re')
    q_squared = qx**2 + qy**2

    sigma = np.zeros_like(Sxx)

    for i in range(sigma.shape[0]):
        for j in range(sigma.shape[1]):
            qx_val = qx[i, j]
            qy_val = qy[i, j]
            q_squared_val = q_squared[i, j]

            if q_squared_val == 0:
                # Handle the q = 0 case separately
                sigma[i, j] = Szz[i, j] + Sxx[i, j] + Syy[i, j]
            else:
                qx2_over_q2 = (qx_val ** 2) / q_squared_val
                qy2_over_q2 = (qy_val ** 2) / q_squared_val
                qxqy_over_q2 = (qx_val * qy_val) / q_squared_val

                sigma[i, j] = (Szz[i, j] + Sxx[i, j] * (1 - qx2_over_q2) 
                               + Syy[i, j] * (1 - qy2_over_q2) 
                               - qxqy_over_q2 * Sxy[i, j] - qxqy_over_q2 * Syx[i, j])

    return qx, qy, sigma
        
def plot_differential_cross_section(data_dir, output_image, cmap="plasma", vmin=None, vmax=None, log_scale=False):
    qx, qy, sigma = calculate_differential_cross_section(data_dir)
    
    if log_scale:
        sigma = np.where(sigma > 0, sigma, np.nan) 
        sigma_log = np.log10(sigma)
        if vmin is None:
            vmin = np.nanmin(sigma_log)  
        if vmax is None:
            vmax = np.nanmax(sigma_log)
    else:
        sigma_log = sigma
        if vmin is None:
            vmin = np.min(sigma)
        if vmax is None:
            vmax = np.max(sigma)
    
    plt.figure(figsize=(8, 6))
    c = plt.pcolor(qx, qy, sigma_log, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(c, label=r'$\log(\sigma)$' if log_scale else r'$\sigma(q)$')
    plt.xlabel(r"$q_{x} \, a / \pi$")
    plt.ylabel(r"$q_{y} \, a / \pi$")
    plt.title(r"Differential Cross Section $\sigma(\mathbf{q})$")
    plt.savefig(output_image, dpi=600)
    plt.close()

#main
if __name__ == "__main__":
    
    # norm_const deals with the polarised state where many S_{ij} are constant and cant be normalised like the others
    # S_{ij} is normalised relative to the max value of S_{zz} ---> after analytical check, this makes norm_const = 1 / Lx^2
    norm_const = 15 ** -2
    
    data_dir = "kd_tree_approach/out_sk"  
    output_image = "kd_tree_approach/out_sk/structure_factors.jpg"
    color_map = "RdBu_r"  
    
    # log_scale maps non-positive values to NaN and windows filled with NaN values will be left transparent
    plot_structure_factors(data_dir, output_image, norm_const, data_type='Norm', cmap=color_map, vmin=0, vmax=1000, log_scale=False)
    
    output_image = "kd_tree_approach/out_sk/cross_section.jpg" 
    plot_differential_cross_section(data_dir, output_image, cmap=color_map, vmin=None, vmax=None, log_scale=True)

    
# if run in REPL from the plots folder
# data_dir = "../out_sk"  # we need to go one step back otherwise the script will look for the out_sk folder inside plots
# output_image = "../out_sk/structure_factors.jpg"  
# plot_structure_factors(data_dir, output_image)
# output_image = "../out_sk/cross_section.jpg"  
