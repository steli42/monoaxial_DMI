import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import simpson  

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

def plot_structure_factors(data_dir, plot_titles, output_image, norm_const, qx_min, qx_max, 
                           data_type='Re', cmap="plasma", vmin=None, vmax=None, log_scale=False): #we assume that the data forms a square grid so qy follows the same limits
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0, hspace=0)

    cmap = cmap

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

        ax.text(qx_min + 0.1, qx_max - 0.1, f"${title}$", color="white", fontsize=24, ha="left", va="top")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([qx_min, qx_max])
        ax.set_ylim([qx_min, qx_max])

        if i >= 6:  # Only label the x-axis of the bottom row
            ax.set_xlabel(r"$q_{x} \, a $",fontsize=20)
        if i % 3 == 0:  # Only label the y-axis of the left column
            ax.set_ylabel(r"$q_{y} \, a $",fontsize=20)

    fig.subplots_adjust(left=0.11, right=0.89, top=0.89, bottom=0.11)  # Add margin around the grid (opposites must add up to 1)

    # Create a common colorbar that spans the entire width of the figure
    cbar_ax = fig.add_axes([0.11, 0.04, 0.775, 0.03])  
    cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal', extend='max')
    cbar.ax.tick_params(labelsize=18)
    plt.text(0.1, 0.04, cbar_label, rotation=0, ha='right', va='bottom', fontsize=20, transform=fig.transFigure)
    
    plt.savefig(output_image, dpi=600)
    plt.close(fig)

def calculate_differential_cross_section(data_dir, S_elements, polarised=False):
    Sxx = read_csv_data(os.path.join(data_dir, S_elements[0] + ".csv"), 'Re')[2]
    Sxy = read_csv_data(os.path.join(data_dir, S_elements[1] + ".csv"), 'Re')[2]
    Syx = read_csv_data(os.path.join(data_dir, S_elements[3] + ".csv"), 'Re')[2]
    Syy = read_csv_data(os.path.join(data_dir, S_elements[4] + ".csv"), 'Re')[2]
    Szz = read_csv_data(os.path.join(data_dir, S_elements[8] + ".csv"), 'Re')[2]

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

                if polarised == False:
                    sigma[i, j] = (Szz[i, j] + 
                                + Sxx[i, j] * (1 - qx2_over_q2) + Syy[i, j] * (1 - qy2_over_q2) 
                                - qxqy_over_q2 * Sxy[i, j] - qxqy_over_q2 * Syx[i, j])
                else: 
                    sigma[i, j] = (Sxx[i, j] * (1 - qx2_over_q2) + Syy[i, j] * (1 - qy2_over_q2) 
                                - qxqy_over_q2 * Sxy[i, j] - qxqy_over_q2 * Syx[i, j])    

    return qx, qy, sigma
       
def plot_differential_cross_section(data_dir, output_image, qx_min, qx_max, S_elements, 
                                    polarised=False, cmap="plasma", vmin=None, vmax=None, log_scale=False):
    qx, qy, sigma = calculate_differential_cross_section(data_dir, S_elements, polarised)
    
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
    plt.xlim([qx_min, qx_max])
    plt.ylim([qx_min, qx_max])
    plt.xlabel(r"$q_{x} \, a $")
    plt.ylabel(r"$q_{y} \, a $")
    if polarised == False:
        plt.title(r"Unpolarised Differential Cross Section $\sigma(\mathbf{q})$")
    else:
        plt.title(r"Polarised Differential Cross Section $\sigma(\mathbf{q})$")
    plt.savefig(output_image, dpi=600)
    plt.close()

def plot_connected_cross_section(data_dir, output_image, qx_min, qx_max, S_elements1, S_elements2, 
                                 polarised=False, cmap="plasma", vmin=None, vmax=None, log_scale=False):
    qx, qy, sigma1 = calculate_differential_cross_section(data_dir, S_elements1, polarised)
    _, _, sigma2 = calculate_differential_cross_section(data_dir, S_elements2, polarised)
    sigma = sigma1 - sigma2

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
    plt.colorbar(c, label=r'$\log(\sigma_Q-\sigma_C)$' if log_scale else r'$\sigma_Q-\sigma_C$')
    plt.xlim([qx_min, qx_max])
    plt.ylim([qx_min, qx_max])
    plt.xlabel(r"$q_{x} \, a $")
    plt.ylabel(r"$q_{y} \, a $")
    if polarised == False:
        plt.title(r"Unpolarised Connected Cross Section $\sigma(\mathbf{q})$")
    else:
        plt.title(r"Polarised Connected Cross Section $\sigma(\mathbf{q})$")
    plt.savefig(output_image, dpi=600)
    plt.close()

def calculate_radial_average(qx, qy, sigma, num_bins):

    # Convert Cartesian coordinates (qx, qy) to polar coordinates (q, theta)
    q = np.sqrt(qx**2 + qy**2)  
    #theta = np.arctan2(qy, qx)  

    # Create bins for radial averaging
    q_max = np.max(qx)
    q_bins = np.linspace(0, q_max, num_bins + 1)
    sigma_radial_avg = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)

    # Iterate over all points and bin the sigma values based on the radial distance q
    for i in range(sigma.shape[0]):
        for j in range(sigma.shape[1]):
            q_val = q[i, j]
            sigma_val = sigma[i, j]

            # Find the bin index for the current q_val
            bin_idx = np.digitize(q_val, q_bins) - 1  # bin_idx starts from 0
            if bin_idx >= 0 and bin_idx < num_bins:
                sigma_radial_avg[bin_idx] += sigma_val
                bin_counts[bin_idx] += 1

    # Normalize by the number of points in each bin
    sigma_radial_avg = np.divide(sigma_radial_avg, bin_counts, where=bin_counts > 0)

    # Return the radial bins (as the centers of the bins) and the averaged sigma values
    q_bin_centers = 0.5 * (q_bins[:-1] + q_bins[1:])
    return q_bin_centers, sigma_radial_avg

def plot_radial_averages(data_dirs, custom_labels, num_bins, S_elements, 
                         polarised=False, normalised=False, log_log=False, output_image="radial_averages.jpg"):
   
    plt.figure(figsize=(8, 6))

    for data_dir,label in zip(data_dirs,custom_labels):
        
        qx, qy, sigma = calculate_differential_cross_section(data_dir, S_elements, polarised=polarised)
        q_bin_centers, sigma_radial_avg = calculate_radial_average(qx, qy, sigma, num_bins)

        if normalised == True:
            sigma_radial_avg /= np.max(sigma_radial_avg)

        label = label  # Use the directory name as a label
        plt.plot(q_bin_centers, sigma_radial_avg, label=label, marker='o', markersize=4, markerfacecolor='none')
               
    if log_log == True:
        plt.xscale('log')
        plt.yscale('log')  

    plt.xlabel(r"$q \, a$")
    plt.ylabel(r"$I(q)$")
    
    if polarised == True:
        plt.title("Radial Average of Polarised Cross Section")
    else:
        plt.title("Radial Average of Unpolarised Cross Section")
    plt.legend()
    plt.grid(True, which='major', linestyle='--')
    # plt.minorticks_on() 

    # Save the figure
    plt.savefig(output_image, dpi=600)
    plt.close()

def calculate_p(q_bin_centers, I_q, r_values):
    p_values = np.zeros_like(r_values)
    for idx, r in enumerate(r_values):
        # Compute the integral using Simpson's rule
        integrand = I_q * np.sin(q_bin_centers * r) * q_bin_centers
        p_values[idx] = r * simpson(integrand, x=q_bin_centers)
    return p_values    
