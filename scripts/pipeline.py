""" Main pipeline file - use for performance testing """

import dask
from dask import annotate
from datetime import datetime
import numpy as np
import os
import pandas as pd
import psutil
import subprocess
import uuid
import time
import sys

from dask import dataframe as ddf
from dask.distributed import Client

from . import preprocess

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
LOCALDATA_DIR = os.path.join(ROOT_DIR, "localdata")
TEMP_DIR = os.path.join(ROOT_DIR, "temp")
COORD_FILE = os.path.join(DATA_DIR, "flywire_synapses_783.parquet")
MAIN_FILE = os.path.join(DATA_DIR, "proofread_connections_783.parquet")


############################## HELPER FUNCTIONS ################################

def get_bytes_per_row(df):
    """ Return the average bytes per row over a sample of the dataframe """
    subdf = df.partitions[0].head(1000) # Sample the first 1000 rows
    bytes_per_row = subdf.memory_usage(deep=True).sum() / len(subdf)
    return bytes_per_row
    
    
def downsample(connectome, num_requested_rows, allow_bytes):
    """ Sample up to n rows from connectome, respecting RAM allowance.
    
    Sample approximately n rows from connectome dataframe if it will stay under 
    RAM allowance when computed, otherwise sample as many rows as possible within
    the RAM allowance (allow_bytes). A hard limit of 0.1 * allow_bytes is used
    to help constrain unmanaged RAM usage - many operations on computed data
    create extra data during their operation.

    Use to keep RAM in check during plotting and statistical analysis, where
    functions cannot operate on dask dataframes directly.
    
    """    
    limiter = 0.1
    
    # Get the amount of RAM the request demands
    bytes_per_row = get_bytes_per_row(connectome)
    request_ram = bytes_per_row * num_requested_rows
    
    # Use hard limit of limiter * allow_bytes for downsampled dataframe to reduce
    # both unmanaged RAM usage and disk spillage
    if request_ram < limiter * allow_bytes:
        # Within limit - sample the dataframe
        row_count = sum(connectome.map_partitions(len).compute())
        if row_count <= num_requested_rows:
            return connectome # Cannot sample more rows than are present
        frac = num_requested_rows / row_count
        sample = connectome.sample(frac=frac)        
        return sample

    # Fallback - warn and return a dataframe as large as limiter * allow_bytes
    multiplier = request_ram / (limiter * allow_bytes)
    capped_num_rows = num_requested_rows / multiplier
    frac = capped_num_rows / row_count
    sample = connectome.sample(frac=frac)    
    new_ram = get_bytes_per_row(sample) * capped_num_rows
    print(f"Number of requested rows ({num_requested_rows}) too large "
          f"({request_ram/(1024**3):.1f} GiB). Using {capped_num_rows} rows instead "
          f"({new_ram/(1024**3):.1f} GiB).")
    return sample
        

def relax_memory_limits():
    """ Let Dask use about half the machine's available RAM. 
    
    On client initialisation, Dask is allocated just over half the machine's
    RAM. This is to help control RAM usage during compute-heavy and memory-
    intensive out-of-dask stages of the pipeline.By default, Dask tries to use 
    roughly half of its allocated RAM. This results in Dask underutilising RAM 
    during Dask-heavy tasks.
    
    This function relaxes Dask's default memory configurations to improve
    allocated RAM utilisation during shuffle-heavy operations.
    
    https://distributed.dask.org/en/stable/worker-memory.html
    
    """
    dask.config.set({"distributed.worker.memory.target": 0.85,
                     "distributed.worker.memory.spill": 0.90,
                     "distributed.worker.memory.pause": 0.95,
                     "distributed.worker.memory.terminate": 0.99})


def restore_memory_limits():
    """ Let Dask use the default worker memory configuration. """
    dask.config.set({"distributed.worker.memory.target": 0.60,
                     "distributed.worker.memory.spill": 0.70,
                     "distributed.worker.memory.pause": 0.80,
                     "distributed.worker.memory.terminate": 0.95})


def make_plot(func_name: str, outpath: str, samples: list, labels: list):
    """ Spawn a process to generate a plot using the specified function 
    
    The input dataframe/s in samples should be adequately downsampled already.
    
    """
    temp_dir_path = make_temp_path()
    
    # Create a subdirectory for every df in samples
    for df, label in zip(samples, labels):
        df_dir_path = os.path.join(temp_dir_path, label)
        df.to_parquet(df_dir_path)
    
    # Call plotting function
    result = subprocess.run(
            ["python", os.path.join(ROOT_DIR, "scripts", "plotting.py"), 
             func_name, temp_dir_path, outpath],
            capture_output = True,
            check = False,
            text = True
    )
    if result.returncode != 0:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")


def make_temp_path():
    """ Generate a temporary file path """
    temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}")
    return temp_path


def clean_temp():
    raise NotImplementedError




##################################### INIT #####################################

def get_ram_allowance():
    """ Program should aim to use about 60% of the available RAM. 
    
    The result of this function is used to determine dask client memory_limit,
    and for guiding compute sizes later (e.g., for statistical analysis).
    """
    mem = psutil.virtual_memory().total # Available RAM in bytes
    mem_gib = mem / (1024**3) # Available RAM in GiB
    allow_bytes = mem * 0.6
    allow_gib = mem_gib * 0.6
    print(f"{mem_gib:.1f} GiB available on machine; allowing {allow_gib:.1f} GiB")
    return allow_bytes


def get_partition_size(num_threads, allow_bytes):
    """ Estimate a safe partition size in bytes.
    
    Dask can be trigger happy - it sees a thread and uses it, even if it means
    going over the RAM allowance. Additionally, shuffles can require 2-3x the
    size of the data being shuffled. This function takes number of threads and 
    the RAM allowance into account to estimate a safe dataframe partition size 
    that is less likely to blow up memory during a big merge.
    
    This doesn't account for concurrency overhead; it is just a buffer function.
    
    """
    max_partition_size = 150 * (1024**2) # 150 MiB
    lower_thresh = 40 * (1024**2) # 40 MiB
    
    partition_size = allow_bytes / (7 * num_threads)
    
    if partition_size > max_partition_size:
        return max_partition_size
    if partition_size < lower_thresh:
        print("Notice: very small partition size "
              f"({partition_size/(1024**2):.1f} MiB)")
        
    return partition_size


def start_dask(num_threads, allow_bytes):
    """ Initialise a dask client. Number of threads = number of cores. """
    # Configure Dask Client using threads because pipeline is data transfer heavy.
    # Capping number of concurrent shuffle tasks to number of threads because 
    # default config caused memory issues (spilling to disk, pausing worker) 
    # due to the scheduler over-allocating tasks with respect to available RAM.
    client = Client(processes=False, 
                    threads_per_worker=num_threads, 
                    memory_limit=allow_bytes,
                    resources={"shuffle": num_threads})
    print(f"\nDask Dashboard at {client.dashboard_link}\n")
    return client




############################## STAGE 1 : LOAD ##################################

def load_connectome(file, partition_size):
    """ Load parquet connectome file into dask dataframe """
    global MAIN_FILE
    if not file: file = MAIN_FILE
    print(f"{datetime.now().strftime("%H:%M:%S")} Loading connectome ...")
    
    meta = {"pre_pt_root_id": np.int32,
            "post_pt_root_id": np.int32,
            "neuropil": "category",
            "gaba_avg": np.float32,
            "ach_avg": np.float32,
            "glut_avg": np.float32,
            "oct_avg": np.float32,
            "ser_avg": np.float32,
            "da_avg": np.float32
    }
    connectome = ddf.read_parquet(
        file, 
        columns = list(meta.keys()),
        meta = meta
    )
    connectome["neuropil"] = connectome["neuropil"].astype(
         "category").cat.as_known()
    connectome = connectome.rename(
        columns = {"pre_pt_root_id": "pre", "post_pt_root_id": "post",
                   "gaba_avg": "gaba", "ach_avg": "ach", "glut_avg": "glut",
                   "oct_avg": "oct", "ser_avg": "ser", "da_avg": "da"})
    connectome = connectome.repartition(partition_size=partition_size).persist()
    
    print(f"{datetime.now().strftime("%H:%M:%S")} Connectome loaded")
    return connectome


def normalise_nt_probs(connectome):
    """ Ensure probabilities sum to 1 for each edge, discarding NaN rows """
    print(f"{datetime.now().strftime("%H:%M:%S")} Normalising neurotransmitter probabilities ...")
    other_cols = ["glut", "oct", "ser", "da"]
    
    # Cast to float64 for increased accuracy in division
    for col in ["gaba", "ach"] + other_cols:
        connectome[col] = connectome[col].astype(np.float64)
    
    # Normalise
    connectome["other"] = connectome[other_cols].sum(axis=1)
    connectome["total_prob"] = connectome[["gaba", "ach", "other"]].sum(axis=1)
    connectome["gaba"] = connectome["gaba"] / connectome["total_prob"]
    connectome["ach"] = connectome["ach"] / connectome["total_prob"]
    connectome["other"] = connectome["other"] / connectome["total_prob"]
    connectome = connectome.drop(columns=["total_prob"])
    connectome = connectome.drop(columns=other_cols)
    connectome = connectome.dropna().persist(subset=["gaba", "ach", "other"]) # Drop NaNs
    
    # Recast to float32 to save RAM
    for col in ["gaba", "ach", "other"]:
        connectome[col] = connectome[col].astype(np.float32)
    
    print(f"{datetime.now().strftime("%H:%M:%S")} Neurotransmitter probabilities normalised")
    return connectome


def load_coord_file(partition_size):
    """ Load in edge coordinate data from 12.7 GB parquet file. 
    It is safe to omit neurotransmitter probabilities because each edge has
    the same neurotransmitter probability, and these probabilities have already
    been normalised in the preceeding normalise_nt_probs step.
    """
    global DATA_DIR
    coord_file_path = os.path.join(DATA_DIR, "flywire_synapses_783.parquet")
    meta = {"pre_pt_root_id": np.int32, 
            "post_pt_root_id": np.int32, 
            "pre_pt_position_x": np.int32, 
            "pre_pt_position_y": np.int32,
            "pre_pt_position_z": np.int32, 
            "post_pt_position_x": np.int32, 
            "post_pt_position_y": np.int32, 
            "post_pt_position_z": np.int32
    }
    edge_coords = ddf.read_parquet( # Don't read in neurotransmitter prob cols!
        coord_file_path, 
        columns = list(meta.keys()), 
        meta = meta
    )
    edge_coords = edge_coords.rename(
        columns={"pre_pt_root_id":"pre", "post_pt_root_id": "post"})
    edge_coords = edge_coords.repartition(partition_size=partition_size).persist()
    return edge_coords


def attach_synapse_coords(connectome, partition_size):
    """ Add x, y, z coordinates for every synaptic connection in connectome.
    
    In the coordinate file, each synapse has a column containing the 'pre' neuron's
    synapse coordinates, and a column containing the 'post' neuron's synapse 
    coordinates.
    
    This function calculates the midpoint x,y,z coordinates for every 
    synapse, and uses the product as the synapse's coordinates.
    
    """
    print(f"{datetime.now().strftime("%H:%M:%S")} Attaching coordinates ...")    
    connectome = connectome.persist()
    coord_df = load_coord_file(partition_size)
    
    # Tag merge tasks with shuffle resource limit to reduce risk of memory 
    # issues. This approach increases stability but reduces concurrency. See
    # start_dask() for more info.
    with annotate(resources = {"shuffle": 1}):
        merged = connectome.merge(
            coord_df, on=["pre","post"], how="inner").reset_index(drop = False)
        
    # Get x, y, z coordinates for each synapse
    merged["x"] = merged["pre_pt_position_x"] + merged["post_pt_position_x"] / 2
    merged["y"] = merged["pre_pt_position_y"] + merged["post_pt_position_y"] / 2
    merged["z"] = merged["pre_pt_position_z"] + merged["post_pt_position_z"] / 2
    merged = merged.drop(columns=["pre_pt_position_x", "post_pt_position_x",
                                  "pre_pt_position_y", "post_pt_position_y",
                                  "pre_pt_position_z", "post_pt_position_z"])
    
    # Coerce floats to ints (not worried about decimal precision)
    merged["x"] = merged["x"].astype(np.int32)
    merged["y"] = merged["y"].astype(np.int32)
    merged["z"] = merged["z"].astype(np.int32)
    
    merged = merged.persist()
    print(f"{datetime.now().strftime("%H:%M:%S")} Coordinates attached")
    return merged


def attach_neuropil_metadata(connectome):
    """ Add high-level neuropil regions and neuropil names to connectome. 
    
    Each neuropil is situated within a higher-level neuropil. For example, the 
    medulla neuropil is part of the optic lobe neuropil. 
    
    The dataset represents neuropils by codes. Attach neuropil names for ease
    of later data interpretation.
    
    """
    print(f"{datetime.now().strftime("%H:%M:%S")} Attaching neuropil metadata ...")
    
    # Load neuropils.csv
    path = os.path.join(LOCALDATA_DIR, "neuropils.csv")
    neuropil_metadata = ddf.read_csv(path)
    neuropil_metadata["neuropil"] = neuropil_metadata["neuropil"].astype(
        "category").cat.as_known()
    
    # Coerce connectome neuropil column to use same categories as metadata df
    cats = neuropil_metadata["neuropil"].cat.categories
    connectome["neuropil"] = connectome["neuropil"].cat.set_categories(cats)
    
    # Merge with connectome on neuropil ID then return
    merged = connectome.merge(neuropil_metadata, on="neuropil", how="inner").persist()
    
    print(f"{datetime.now().strftime("%H:%M:%S")} Neuropil metadata attached")
    return merged




########## STAGE 2 : IDENTIFY NEUROTRANSMITTER PROBABILITY CLUSTERS ##########


def tag_clusters(connectome):
    """ Attach column with initial neurotransmitter cluster assignment """

    def _tag(pdf):
        tags = pd.Series("none", index=pdf.index)
        tags[(pdf["gaba"] > 0.85) & (pdf["ach"] < 0.1)] = "gaba"
        tags[(pdf["other"] > 0.85) & (pdf["ach"] < 0.1) &
             (pdf["gaba"] < 0.1)] = "other"
        tags[(pdf["gaba"] < 0.15) & (pdf["other"] < 0.45) &
             (pdf["ach"] > 0.55)] = "ach"

        return tags

    tags = connectome.map_partitions(_tag, meta=("cluster", "category"))
    return connectome.assign(cluster=tags).persist()


def tag_unclassified_clusters(connectome):
    """ Update 'none' values in cluster column with sub-classifications """
    # De-categorise old cluster values in metadata AND graph (fails without
    # _meta update)
    connectome = connectome.map_partitions(
        lambda pdf: pdf.assign(cluster=pdf["cluster"].astype("object")),
        meta=connectome._meta.assign(cluster="object"))
    
    def _tag(pdf):
        tags = pd.Series(pd.NA, index=pdf.index)
        mask_none = pdf["cluster"] == "none"
        tags[mask_none & 
             (pdf["ach"] < 0.15) & 
             (pdf["gaba"] > 0.1)] = "low_ach_noise"
        tags[mask_none & 
             (pdf["gaba"] <= 0.15) & 
             (pdf["ach"] >= 0.1)] = "low_gaba_noise"
        return tags
    
    tags = connectome.map_partitions(_tag, meta=("cluster", "category"))
    updated = connectome.assign(
        cluster = connectome["cluster"].where(tags.isna(), tags))
    updated["cluster"] = updated["cluster"].astype("category").cat.as_known()
    
    return updated.persist()


def make_brain_map(sample_path: str, outdir: str):
    """ Spawn a subprocess that generates an interactive 3D brain map """
    result = subprocess.run(
        ["python", os.path.join(ROOT_DIR, "scripts", "brainz.py"), 
         sample_path, outdir],
        capture_output = True,
        check = False,
        text = True
    )
    if result.returncode != 0:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")        




######################## STAGE 3 : NEUROPIL VISUALISATIONS #####################


def get_neuropil_summary_stats(connectome):
    """ Aggregate synapse data by neuropil. 
    For each neuropil, get synapse count, average neurotransmitter probabilities,
    and neurotransmitter probability variances.
    """
    print(f"{datetime.now().strftime("%H:%M:%S")} Aggregating neuropil data ...")
    trimmed = connectome.drop(columns = ["pre","post","x","y","z"])
    # Get aggregated data (neurotransmitter means and variances)
    agg_dict = {"gaba": ["mean", "var"],
                "ach": ["mean", "var"],
                "other": ["mean", "var"]}
    aggregated = trimmed.groupby(["neuropil"]).agg(agg_dict)
    aggregated.columns = [f"{col}_{stat}" if stat else col
        for col, stat in aggregated.columns.to_flat_index()]
    aggregated = aggregated.reset_index(drop=False).set_index(
        "neuropil", drop=True).persist()
    
    # Get synapse counts per neuropil
    counts = trimmed.groupby(["neuropil"]).count().reset_index(
        drop=False).set_index("neuropil", drop=True)
    counts = counts.drop(columns = ["gaba", "ach"]).rename(
        columns={"other": "neuropil_size"}).persist()
    
    # Combine aggregated data into one dataframe
    merged_aggregated = counts.merge(aggregated, on=["neuropil"], how="inner")
    merged_aggregated = merged_aggregated.set_index("neuropil", drop=False).persist()
    print(f"{datetime.now().strftime("%H:%M:%S")} Neuropil data aggregated")
    return merged_aggregated



########################### STAGE 4 : STATISTICS ###############################


def do_stats(r_path, result_dir, allow_bytes):
    """ Spawn a subprocess that runs the R stats script 
    
    The input dataframe should be downsampled already.
    
    """
    # Write dataframe to parquet dataset folder in temp directory    
    temp_path = make_temp_path()
    df.to_parquet(temp_path)
    
    # Call R script
    result = subprocess.run(
        ["python", os.path.join(ROOT_DIR, "scripts", "statistical_analysis.R"),
         temp_file_path, outdir],
        capture_output = True,
        check = True,
        text = True
    )



################################### MAIN #######################################

def main(num_threads, dataset):
    """ This section mimics the Jupyter Notebook code """    
    session_id = f"{dataset.strip(".parquet")}_{num_threads}_threads"
    OUTDIR = os.path.join(ROOT_DIR, "results", session_id)
    start_time = time.time()
    
    allow_bytes = get_ram_allowance()
    client = start_dask(num_threads=num_threads, allow_bytes=allow_bytes)
    PARTITION_SIZE = get_partition_size(num_threads = num_threads, allow_bytes = allow_bytes)    
    dask.config.set({"dataframe.shuffle.method": "tasks"})
    
    # Processing
    relax_memory_limits()
    connectome = load_connectome(dataset, PARTITION_SIZE)
    connectome = normalise_nt_probs(connectome)
    connectome = attach_synapse_coords(connectome, PARTITION_SIZE)
    connectome = attach_neuropil_metadata(connectome)
    restore_memory_limits()
        
    connectome_nts = connectome[["gaba","ach","other"]]
    
    # Sample connectome - hexbin plotting is memory-intensive and don't need every data point to get idea of distribution    
    sample = pipeline.downsample(connectome_nts, 50_000, allow_bytes)
    filename = os.path.join(OUTDIR, "init_overall_distribution.png")
    pipeline.make_plot("plot_n_ternary", filename, samples = [sample], labels = ["Neurotransmitter_Probabilities_In_The_Drosophila_Connectome"])
    
    # Extract and visualise the four initial clusters
    connectome = pipeline.tag_clusters(connectome)
    sample = pipeline.downsample(connectome, 50_000, allow_bytes)
    
    # Plot hex plots of new cluster classifications
    grouped = sample.groupby("cluster")
    high_gaba_sample = grouped.get_group("gaba")
    high_ach_sample = grouped.get_group("ach")
    high_other_sample = grouped.get_group("other")
    unclassified_sample = grouped.get_group("none")
    
    filename = os.path.join(OUTDIR, "initial_clusters.png")
    pipeline.make_plot("plot_n_ternary", filename, 
                       samples = [high_gaba_sample, high_ach_sample, high_other_sample, unclassified_sample], 
                       labels = ["High_GABA", "High_ACH", "High_OTHER", "Unclassified"])    
    
    # Classify unclassified synapses into gaba_noise, ach_noise, and noise.
    # Taking a larger sample this time because I want more power in my analysis.
    connectome = pipeline.tag_unclassified_clusters(connectome)
    print("All synapses tagged!")
    sample = pipeline.downsample(connectome, 500_000, allow_bytes)    
    # Plot hex plots of new cluster classifications
    grouped = sample.groupby("cluster")
    high_gaba_sample = grouped.get_group("gaba")
    high_ach_sample = grouped.get_group("ach")
    high_other_sample = grouped.get_group("other")
    low_ach_noise_sample = grouped.get_group("low_ach_noise")
    low_gaba_noise_sample = grouped.get_group("low_gaba_noise")
    misc_noise_sample = grouped.get_group("misc_noise")
    
    filename = os.path.join(OUTDIR, "final_clusters.png")
    pipeline.make_plot("plot_n_ternary", filename, 
                       samples = [high_gaba_sample, high_ach_sample, high_other_sample, 
                                  low_ach_noise_sample, low_gaba_noise_sample, misc_noise_sample], 
                       labels = ["High_GABA", "High_ACH", "High_OTHER",
                                 "Low_ACH_Noise", "Low_GABA_Noise", "Misc_Noise"])
    
    # Skipping brain map section because that flunked
    
    # Summary stats per neuropil
    neuropils = pipeline.get_neuropil_summary_stats(connectome)
    filename = os.path.join(OUTDIR, "neuropil_summaries.csv")
    neuropils.to_csv(filename)

    
    # Neuropil neurotransmitter probability distributions by region
    grouped_regions = sample.groupby("region")
    regions = list(sample["region"].unique().compute())
    for region in regions:
        df = grouped_regions.get_group(region)
        filename = os.path.join(OUTDIR, f"neuropil_distributions_{region}.png")
        pipeline.make_plot("plot_neuropils_in_region", filename,
                          samples = [df], labels = [region])
        
        

    # End timer
    end_time = time.time()
    print(f"Runtime: {end - start:.2f} seconds; Num-threads: {num_threads}")
    
    
    
def perf_test(num_threads, dataset_code):
    """ Performance test main pipeline. 
    Note: preprocess.py must already be executed """
        
    # Set up environment
    if dataset_code == 0:
        DATASET = os.path.join(DATA_DIR, "tiny.parquet") # ~ 215 KB
    elif dataset_code == 1:
        DATASET = os.path.join(DATA_DIR, "small.parquet") # ~ 56 MB
    elif dataset_code == 2:
        DATASET = os.path.join(DATA_DIR, "medium.parquet") # ~ 272 MB
    elif dataset_code == 3:
        DATASET = os.path.join(DATA_DIR, "large.parquet") # ~ 583 MB
    elif dataset_code == 4:
        DATASET = os.path.join(DATA_DIR, "proofread_connections_783.parquet") # ~1.05 GB        
        
    main(num_threads, DATASET)