""" Preprocess raw data 

Convert raw data from feather files to parquet files, and generate scalability 
test files of varying problem sizes.

This script should only need to be ran once.

"""
import dask
import os
import pyarrow
import pyarrow.feather as feather
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
from dask import dataframe as ddf
from dask.distributed import Client
from dask.distributed import get_client
from datetime import datetime

CLIENT = None # Assigned in run()

ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MAIN_FILE_RAW = os.path.join(DATA_DIR, "proofread_connections_783.feather")
COORD_FILE_RAW = os.path.join(DATA_DIR, "flywire_synapses_783.feather")
MAIN_FILE = os.path.join(DATA_DIR, "proofread_connections_783.parquet")
COORD_FILE = os.path.join(DATA_DIR, "flywire_synapses_783.parquet")

METADATA_FILE = os.path.join(DATA_DIR, "metadata.txt")


############################### PREPROCESS #####################################


def _feather_to_parquet(file_to_convert, destination):
    """ Convert feather file to parquet via record batch streaming.
    
    This approach minimises peak RAM, with peak RAM usage roughly proportional
    to the largest record batch in the feather file. This is superior to loading
    an entire 9.5 GB feather file into RAM using the naive write_tables method.
    """
    print(f"{datetime.now().strftime("%H:%M:%S")} Converting {file_to_convert} "
          "to parquet file ...")
    
    # Create streaming reader for random file access (no load!)
    reader = pyarrow.ipc.open_file(file_to_convert) 
    writer = None
    chunk_size = 250_000    
    
    # Iteratively load, convert, and store feather record batches to parquet
    for i in range(reader.num_record_batches):
        batch = reader.get_record_batch(i) # Get a subset of rows from feather
        if i == 0: # Print batch info for the first batch
            print(f"Record batch contains {batch.num_rows} rows "
                  f"({batch.nbytes / (1024**3)} GB)")
        table = pa.Table.from_batches([batch]) # Convert batch to pyarrow table
        
        # Create parquet writer (once). The writer needs a schema, and that 
        # schema is supplied by the table, hence cannot do this step outside
        # the for loop.
        if writer is None:
            writer = pq.ParquetWriter(destination, table.schema)
            
        # Write batch into row groups of chunk size. Chunk size of 250,000
        # means 250,000 rows per row group/chunk. Currently do not know how
        # large the batches are in the feather file but they could be much
        # larger than 250,000. If they are, it will dramatically slow down
        # the pipeline. 250,000 rows per chunk seems to be a healthy middle ground.
        writer.write_table(table, row_group_size = chunk_size)
        
    writer.close()
    print(f"{datetime.now().strftime("%H:%M:%S")} {file_to_convert} converted "
          "to parquet file")


def _get_all_node_ids(df, node_cols=["pre","post"]):
    """ Return a dataframe containing every unique node id in the dataframe """
    node_cols = [df[col].rename("node_id").to_frame() for col in node_cols]
    all_nodes = ddf.concat(node_cols).drop_duplicates().repartition(npartitions=1).reset_index(drop=True)
    return all_nodes


def _write_test_metadata(sub_connectome, test_id):
    """ Write test connectome file metadata
    E.g., number of nodes, number of edges, node:edge ratio, neuropils """
    global METADATA_FILE    
    num_nodes = _get_all_node_ids(sub_connectome, 
        node_cols=["pre_pt_root_id","post_pt_root_id"]).count()
    num_edges = sub_connectome["pre_pt_root_id"].count()
    neuropils = sub_connectome["neuropil"].unique()
    
    num_nodes, num_edges, neuropils = dask.compute(
        num_nodes, num_edges, neuropils)
    node_edge_ratio = num_nodes/num_edges
    num_nodes, node_edge_ratio = num_nodes.item(), node_edge_ratio.item()
    
    with open(METADATA_FILE, "a") as mfile:
        mfile.write(f"Testfile Metadata: {test_id}.feather\n")
        mfile.write(f"Number of nodes: {num_nodes}\n")
        mfile.write(f"Number of edges: {num_edges}\n")
        mfile.write(f"Node-edge ratio: {node_edge_ratio}\n")
        mfile.write(f"Neuropils: {list(neuropils)}\n\n")


def _write_subset_file(sub_connectome, test_id):
    """ Write parquet file containing sub-connectome. 
    These tests should fit in memory so will compute to pandas first to avoid
    saving them in their own directories """
    global DATA_DIR
    filename = os.path.join(DATA_DIR, f"{test_id}.parquet")
    pd_sub_connectome = sub_connectome.compute()
    pd_sub_connectome.to_parquet(filename, index=False, compression=None)
    
    
def _make_tests():
    """ Create subsets of main file of varying sizes for performance analysis """
    global MAIN_FILE
    print(f"{datetime.now().strftime("%H:%M:%S")} Generating test parquet files ...")
    connectome = ddf.read_parquet(MAIN_FILE)
    connectome = connectome.categorize(columns=["neuropil"]).persist()    
    
    # Subset dataframe into different sizes
    tiny = connectome.map_partitions(
        lambda df: df[df["neuropil"].str.startswith("BU_")])
    small = connectome.map_partitions(
        lambda df: df[df["neuropil"].str.startswith("LOP_")])   
    medium = connectome.map_partitions(
        lambda df: df[df["neuropil"].str.startswith("ME_")])    
    omit = {"ME_L", "ME_R", "LOP_L", "LOP_R"}
    large = connectome.map_partitions(
        lambda df: df[~df["neuropil"].isin(omit)])
    
    # Write subset data 
    subsets = [tiny, small, medium, large]
    test_ids = ["tiny", "small", "medium", "large"]
    futures = []
    for sub_connectome, test_id in zip(subsets, test_ids):
        futures.append(CLIENT.submit(_write_subset_file, sub_connectome, test_id))
        _write_test_metadata(sub_connectome, test_id)
    CLIENT.gather(futures)
    print(f"{datetime.now().strftime("%H:%M:%S")} Generated test parquet files")


################################### MAIN #######################################

def _is_downloaded(file1, file2):
    if os.path.exists(file1) and os.path.exists(file2):
        return True
    return False


def _all_tests_exist():
    global DATA_DIR
    test_ids = ["tiny", "small", "medium", "large"]
    test_files = [os.path.join(DATA_DIR, f"{_id}.parquet") for _id in test_ids]
    all_exist = all(os.path.exists(file) for file in test_files)
    return True if all_exist else False


def run():
    """ Convert raw data """
    global MAIN_FILE_RAW, COORD_FILE_RAW, CLIENT
    print("Notice: this may take about 10 minutes if this is the first time "
          "running preprocess.py. ")
    
    # Convert feather files to parquet if not already converted
    if not _is_downloaded(MAIN_FILE_RAW, COORD_FILE_RAW):
        raise Exception("Cannot find downloaded feather files in root/data/")
    if not os.path.exists(MAIN_FILE):
        _feather_to_parquet(file_to_convert=MAIN_FILE_RAW, destination=MAIN_FILE)
    if not os.path.exists(COORD_FILE):
        _feather_to_parquet(file_to_convert=COORD_FILE_RAW, destination=COORD_FILE)
    
    # Make test parquet files of varying sizes
    CLIENT = get_client()
    if not _all_tests_exist():
        _make_tests()
    
    print(f"\n{datetime.now().strftime("%H:%M:%S")} Preprocessing complete!")