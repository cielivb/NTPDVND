""" Make ternary graphs. 
Most plots will be hexbin heatmaps due to large number of points.
"""

import math
import matplotlib.pyplot as plt
import mpltern
import numpy as np
import os
import pandas as pd
import sys



def _make_ternary_subplots(n: int):
    """ Create a matplotlib plot with n ternary subplots """
    num_cols = math.ceil(math.sqrt(n))
    num_rows = math.ceil(n / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5,5),
                             subplot_kw=dict(projection="ternary"))
    
    # Prevent ternary plots from overlapping
    fig.subplots_adjust(
        left=0.20,
        right=0.25,
        top=0.25,
        bottom=0.20,
        wspace=1,
        hspace=3
    )
    
    axes = np.array(axes).reshape(-1)
    
    for ax in axes.flatten():
        ax.tick_params(labelsize=6) # Make the tick labels smaller
        
        # Make the 'GABA', 'ACH', 'OTHER' labels smaller
        ax.set_tlabel(ax.get_tlabel(), fontsize=8)
        ax.set_llabel(ax.get_llabel(), fontsize=8)
        ax.set_rlabel(ax.get_rlabel(), fontsize=8)        
    
    return fig, axes


def plot_n_ternary(data, outpath):
    """ Plot multiple ternary plots in a single image """
    print("Plotting ...")
    num_plots = len(data)
    fig, axes = _make_ternary_subplots(num_plots)
    
    for i, ax in enumerate(axes[:num_plots]):
        print(data[i][0])
        # Compute dataframe i's probabilities into numpy arrays
        df = data[i][1][["gaba", "ach", "other"]]
        gaba, ach, other = df.to_numpy(copy=False).T
        
        # Plot hexbin
        hb = ax.hexbin(gaba, ach, other, gridsize=20)
        ax.set_tlabel("GABA")
        ax.set_llabel("ACH")
        ax.set_rlabel("OTHER")
        ax.taxis.set_label_position("tick1")
        ax.laxis.set_label_position("tick1")
        ax.raxis.set_label_position("tick1")
        ax.set_title(data[i][0], size = 11, pad = 10)
    
    # Hide unused axes
    for j in range(num_plots, len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout() # fit subplots to figure size
    print("Saving ...")
    plt.savefig(outpath, format="png", dpi=300)


def plot_mean_nt_probs_by_neuropil_size(data, outfile):
    """ Create one ternary scatter plot where point colour corresponds to number 
    of synapses in a neuropil, and each point is the average probability within a
    neuropil. connectome dataframe must be pre-aggregated. """
    num_plots = len(data)
    fig, axes = _make_ternary_subplots(num_plots)
    
    neuropil_size, gaba, ach, other = data[0][1][
        ["neuropil_size", "gaba_mean", "ach_mean", "other_mean"]
        ].compute().to_numpy().T
    
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(projection="ternary")
    scattered = ax.scatter(gaba, ach, other, c=neuropil_size)
    ax.set_tlabel("GABA Probability")
    ax.set_llabel("Acetylcholine Probability")
    ax.set_rlabel("Other Neurotransmitter Probability")
    ax.taxis.set_label_position("tick1")
    ax.laxis.set_label_position("tick1")
    ax.raxis.set_label_position("tick1")
    ax.set_title("Mean Neurotransmitter Probabilities by Neuropil Size", 
                 size=14, pad=50)
    colour_bar = fig.colorbar(scattered, shrink=0.7)
    colour_bar.ax.set_title("Synapse Count", size=10)
    plt.tight_layout()
    plt.savefig(outfile, format="png", dpi=300)


def plot_variance_by_neuropil_size(connectome, outfile):
    """ Create a bar chart showing neuropil neurotransmitter variances, sorted 
    by neuropil size """
    sorted_connectome = connectome.sort_values("neuropil_size").reset_index(drop=False)
    neuropil, gaba, ach, other = sorted_connectome[
        ["neuropil", "gaba_var", "ach_var", "other_var"]].to_numpy().T
    
    # Set bar positions
    bar_width = 0.25
    r = np.arange(len(gaba))
    r2 = r + bar_width
    r3 = r2 + bar_width
    
    fig, ax = plt.subplots(figsize=(8,6), dpi=300)
    ax.bar(r, gaba, color="blue", width=bar_width, label="GABA")
    ax.bar(r2, ach, color="yellow", width=bar_width, label="ACH")
    ax.bar(r3, other, color="pink", width=bar_width, label="OTHER")
    ax.set_xlabel("Neuropil")
    ax.set_xticks(r2)
    ax.set_xticklabels(neuropil)
    ax.legend(title="Neurotransmitter Probability Variances", ncol=3)
    plt.tight_layout()
    plt.savefig(outfile, format="png", dpi=300)


def plot_hex_per_neuropil(connectome, file: str):
    """ Create ternary hexbin plots showing the overall distributions of actual
    probabilities for all synapses in the dask dataframes in connectomes list """
    neuropils = list(connectome["neuropil"].unique().compute())
    grouped = connectome.groupby("neuropil")
    num_plots = len(neuropils)
    
    fig, axes = _make_ternary_subplots(num_plots)
    
    for i, ax in enumerate(axes[:num_plots]):
        
        # Compute neuropil i's probabilities into a numpy array
        group = grouped.get_group(neuropils[i])
        gaba, ach, other = group[["gaba", "ach", "other"]].compute().to_numpy().T
        
        # Plot hexbin
        hb = ax.hexbin(gaba, ach, other, gridsize=20)
        ax.set_tlabel("GABA")
        ax.set_llabel("ACH")
        ax.set_rlabel("OTHER")
        ax.taxis.set_label_position("tick1")
        ax.laxis.set_label_position("tick1")
        ax.raxis.set_label_position("tick1")
        ax.set_title(neuropils[i])
    
    # Hide unused axes
    for j in range(num_plots, len(axes)):
        axes[j].axis("off")
    plt.tight_layout() # fit subplots to figure size
    plt.savefig(file, format="svg", dpi=300)



def plot_neuropils_in_region(data, region, outpath):
    """ Generate hex bin plots for each neuropil in region """
    neuropils = list(data["neuropil"].unique())
    grouped = data.groupby("neuropil")
    num_plots = len(neuropils)
    fig, axes = _make_ternary_subplots(num_plots)
    fig.set_title(region)
    
    for i, ax in enumerate(axes[:num_plots]):
        
        # Compute neuropil i's probabilities into a numpy array
        group = grouped.get_group(neuropils[i])
        gaba, ach, other = group[["gaba", "ach", "other"]].to_numpy().T
        
        # Plot hexbin
        hb = ax.hexbin(gaba, ach, other, gridsize=20)
        ax.set_tlabel("GABA")
        ax.set_llabel("ACH")
        ax.set_rlabel("OTHER")
        ax.taxis.set_label_position("tick1")
        ax.laxis.set_label_position("tick1")
        ax.raxis.set_label_position("tick1")
        ax.set_title(neuropils[i])
    
    # Hide unused axes
    for j in range(num_plots, len(axes)):
        axes[j].axis("off")
    plt.tight_layout() # fit subplots to figure size
    plt.savefig(outpath, format="png", dpi=300)    





def load_parquet_files(inpath):
    """ Return a list of tuples containing formatted labels and respective dask 
    dataframes 
    """
    data = []
    df_folders = os.listdir(inpath)
    for folder_name in df_folders:
        folder_path = os.path.join(inpath, folder_name)
        df = pd.read_parquet(folder_path)
        label = os.path.basename(folder_path).removesuffix(".parquet").replace("_", " ")
        print(f"New label: {label}")
        data.append((label, df))
    return data


# Manage subprocess calls
if __name__ == "__main__":
    
    print("Received call")
    inpath, outpath = sys.argv[2], sys.argv[3]
    print("Loading data from file")
    data = load_parquet_files(inpath)
    print("Routing ...")
    
    if sys.argv[1] == "plot_overall_distribution":
        pdf = data[0][1]
        plot_overall_distribution(pdf, outpath)
        
    elif sys.argv[1] == "plot_n_ternary":
        plot_n_ternary(data, outpath)
    
    elif sys.argv[1] == "plot_mean_nt_probs_by_neuropil_size":
        neuropil_summary_pdf = data[0][1]
        plot_mean_nt_probs_by_neuropil_size(neuropil_summary_pdf, outpath)
        
    elif sys.argv[1] == "plot_neuropils_in_region":
        region, pdf = data[0][0], data[0][1] 
        plot_neuropils_in_region(pdf, region, outpath)