import time
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from sans_aux import (
    plot_structure_factors,
    plot_differential_cross_section,
    plot_connected_cross_section,
    plot_radial_averages,
)
from multiprocessing import Pool, cpu_count

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"


def process_folder(args):

    data_dir, labels, show_ylabel = args 
    
    config_dir = os.path.join(".", "config_local.json")
    with open(config_dir, "r") as file:
        p = json.load(file)

    # norm_const deals with the polarised state where many S_{ij} are constant and cant be normalised like the others
    # S_{ij} is normalised relative to the max value of S_{zz} ---> after analytical check, this makes norm_const =  = 1/Area = 1 / Lx^2
    norm_const = (p["Lx"] * p["Ly"]) ** -2
    qx_max = p["q_max"]
    qx_min = -qx_max
    color_map = "magma"  # "inferno" #"RdBu_r"
    format = "jpg"

    indices = ["x", "y", "z"]
    plot_titles_Q = [f"S_{{{i}{j}}}" for i in indices for j in indices]

    indices = ["x", "y", "z"]
    plot_titles_C = [f"G_{{{i}{j}}}" for i in indices for j in indices]

    
    # log_scale maps non-positive values to NaN and windows filled with NaN values will be left transparent
    # output_image = os.path.join(data_dir, "Q_structure_factors."+format)
    # plot_structure_factors(
    #     data_dir,
    #     plot_titles_Q,
    #     output_image,
    #     norm_const,
    #     qx_min,
    #     qx_max,
    #     data_type="Norm",
    #     cmap=color_map,
    #     vmax=1250
    # )
    
    # output_image = os.path.join(data_dir, "C_structure_factors."+format)
    # plot_structure_factors(
    #     data_dir,
    #     plot_titles_C,
    #     output_image,
    #     norm_const,
    #     qx_min,
    #     qx_max,
    #     data_type="Norm",
    #     cmap=color_map,
    #     vmin=None,
    #     vmax=None,
    #     log_scale=False,
    # )


    output_image = os.path.join(data_dir, "Q_pol_cross_section."+format)
    plot_differential_cross_section(
        data_dir,
        output_image,
        qx_min,
        qx_max,
        S_elements=plot_titles_Q,
        labels=labels,
        show_ylabel=show_ylabel,
        polarised=True,
        cmap=color_map
    )


    # output_image = os.path.join(data_dir, "C_pol_cross_section."+format)
    # plot_differential_cross_section(
    #     data_dir,
    #     output_image,
    #     qx_min,
    #     qx_max,
    #     S_elements=plot_titles_C,
    #     labels=labels,
    #     show_ylabel=show_ylabel,
    #     polarised=True,
    #     cmap=color_map,
    # )

    output_image = os.path.join(data_dir, "pol_connected_cross_section."+format)
    plot_connected_cross_section(
        data_dir,
        output_image,
        qx_min,
        qx_max,
        S_elements1=plot_titles_Q,
        S_elements2=plot_titles_C,
        labels=labels,
        show_ylabel=show_ylabel,
        polarised=True,
        cmap=color_map
    )


if __name__ == "__main__":

    base = os.path.join("..","data_corr", "out_corr")

    # this finds all subfolders in out_corr
    data_dirs = sorted([
        os.path.join(base, d)
        for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d))
    ])
    
    
    label_list = [
    [r"$\rm (a)$", r"$\rm (a)$", r"$c=0$"],
    [r"$\rm (b)$", r"$\rm (b)$", r"$c=1/\sqrt{2}$"],
    [r"$\rm (c)$", r"$\rm (c)$", r"$c=1$"]
    ]

    if len(label_list) != len(data_dirs):
        raise ValueError("Number of labels must match number of data directories.")

    
    args_list = []
    for i, (data_dir, labels) in enumerate(zip(data_dirs, label_list)):
        show_ylabel = (i == 0)   # only the first panel gets a y-label
        args_list.append((data_dir, labels, show_ylabel))


    start_time = time.time()
     
    # num_workers = min(cpu_count(), len(args_list))
    # with Pool(num_workers) as pool:
    #     pool.map(process_folder, args_list)  
        
        
    indices = ["x", "y", "z"]
    plot_titles_Q = [f"S_{{{i}{j}}}" for i in indices for j in indices]

        
    custom_labels = [r"$c=0.0$", r"$c=1/\sqrt{2}$", r"$c=1.0$"]
    plot_radial_averages(
        data_dirs,
        custom_labels,
        S_elements=plot_titles_Q,
        num_bins=195,
        polarised=True,
        normalised=False,
        xlog=True,
        output_image=os.path.join(base, "Q_pol_radial_averages.pdf"),
    )

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
