import time
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from sans_aux import plot_structure_factors
from sans_aux import plot_differential_cross_section
from sans_aux import plot_connected_cross_section
from sans_aux import plot_radial_averages

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

if __name__ == "__main__":

    config_dir = os.path.join("kd_tree_approach", "config.json")
    with open(config_dir, "r") as file:
        p = json.load(file)

    # norm_const deals with the polarised state where many S_{ij} are constant and cant be normalised like the others
    # S_{ij} is normalised relative to the max value of S_{zz} ---> after analytical check, this makes norm_const =  = 1/Area = 1 / Lx^2
    norm_const = (p["Lx"] * p["Ly"]) ** -2
    qx_max = p["q_max"]
    qx_min = -qx_max
    color_map = "magma"  # "inferno" #"RdBu_r"
    base = os.path.join("data_corr", "out_corr")

    plot_titles_Q = [
        "S_{xx}",
        "S_{xy}",
        "S_{xz}",
        "S_{yx}",
        "S_{yy}",
        "S_{yz}",
        "S_{zx}",
        "S_{zy}",
        "S_{zz}",
    ]
    plot_titles_C = [
        "G_{xx}",
        "G_{xy}",
        "G_{xz}",
        "G_{yx}",
        "G_{yy}",
        "G_{yz}",
        "G_{zx}",
        "G_{zy}",
        "G_{zz}",
    ]

    # this finds all subfolders in out_corr...
    data_dirs = [
        os.path.join(base, d)
        for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d))
    ]

    start_time = time.time()
    for data_dir in data_dirs:
        # log_scale maps non-positive values to NaN and windows filled with NaN values will be left transparent
        output_image = os.path.join(data_dir, "Q_structure_factors.jpg")
        plot_structure_factors(
            data_dir,
            plot_titles_Q,
            output_image,
            norm_const,
            qx_min,
            qx_max,
            data_type="Norm",
            cmap=color_map,
            vmin=0,
            vmax=1000,
            log_scale=False,
        )

        output_image = os.path.join(data_dir, "Q_unpol_cross_section.jpg")
        plot_differential_cross_section(
            data_dir,
            output_image,
            qx_min,
            qx_max,
            S_elements=plot_titles_Q,
            polarised=False,
            cmap=color_map,
            vmin=None,
            vmax=None,
            log_scale=True,
        )

        output_image = os.path.join(data_dir, "Q_pol_cross_section.jpg")
        plot_differential_cross_section(
            data_dir,
            output_image,
            qx_min,
            qx_max,
            S_elements=plot_titles_Q,
            polarised=True,
            cmap=color_map,
            vmin=None,
            vmax=None,
            log_scale=False,
        )

        output_image = os.path.join(data_dir, "C_structure_factors.jpg")
        plot_structure_factors(
            data_dir,
            plot_titles_C,
            output_image,
            norm_const,
            qx_min,
            qx_max,
            data_type="Norm",
            cmap=color_map,
            vmin=None,
            vmax=None,
            log_scale=False,
        )

        output_image = os.path.join(data_dir, "C_unpol_cross_section.jpg")
        plot_differential_cross_section(
            data_dir,
            output_image,
            qx_min,
            qx_max,
            S_elements=plot_titles_C,
            polarised=False,
            cmap=color_map,
            vmin=None,
            vmax=None,
            log_scale=False,
        )

        output_image = os.path.join(data_dir, "C_pol_cross_section.jpg")
        plot_differential_cross_section(
            data_dir,
            output_image,
            qx_min,
            qx_max,
            S_elements=plot_titles_C,
            polarised=True,
            cmap=color_map,
            vmin=None,
            vmax=None,
            log_scale=False,
        )

        output_image = os.path.join(data_dir, "pol_connected_cross_section.jpg")
        plot_connected_cross_section(
            data_dir,
            output_image,
            qx_min,
            qx_max,
            S_elements1=plot_titles_Q,
            S_elements2=plot_titles_C,
            polarised=True,
            cmap=color_map,
            vmin=None,
            vmax=None,
            log_scale=True,
        )

        output_image = os.path.join(data_dir, "unpol_connected_cross_section.jpg")
        plot_connected_cross_section(
            data_dir,
            output_image,
            qx_min,
            qx_max,
            S_elements1=plot_titles_Q,
            S_elements2=plot_titles_C,
            polarised=False,
            cmap=color_map,
            vmin=None,
            vmax=None,
            log_scale=True,
        )

    custom_labels = [r"$c=0.0$", r"$c=1/\sqrt{2}$", r"$c=1.0$"]
    plot_radial_averages(
        data_dirs,
        custom_labels,
        num_bins=160,
        S_elements=plot_titles_Q,
        polarised=True,
        normalised=False,
        log_log=True,
        output_image=os.path.join(base, "Q_pol_radial_averages.jpg"),
    )

    plot_radial_averages(
        data_dirs,
        custom_labels,
        num_bins=160,
        S_elements=plot_titles_Q,
        polarised=False,
        normalised=False,
        log_log=True,
        output_image=os.path.join(base, "Q_unpol_radial_averages.jpg"),
    )

    plot_radial_averages(
        data_dirs,
        custom_labels,
        num_bins=160,
        S_elements=plot_titles_C,
        polarised=True,
        normalised=False,
        log_log=True,
        output_image=os.path.join(base, "C_pol_radial_averages.jpg"),
    )

    plot_radial_averages(
        data_dirs,
        custom_labels,
        num_bins=160,
        S_elements=plot_titles_C,
        polarised=False,
        normalised=False,
        log_log=True,
        output_image=os.path.join(base, "C_unpol_radial_averages.jpg"),
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")
