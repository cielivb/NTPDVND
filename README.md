# NTPDVND - Neurotransmitter Probability Distribution Variance Across Neuropils In Drosophila

## Replicate My Results!

**⚠️ DISCLAIMER**: I have only tested this on Windows 11 with 16 GB RAM. I originally intended on making this a cloud computing project (thus OS-independent), but did not wish to pay $0.27 per month to host the large file on the cloud. The data is small enough for many machines to handle locally, so the admin is not worth it.

### Steps Summary
0. Prerequisites
1. Clone repo
2. Download the raw data files into the project's data directory
3. Set up environment
4. Run `jupyter notebook`, then open either the exploratory notebook or performance testing notebook.

### 0. Prerequisites
#### 0.1 Install conda (required for true replication)

This project uses conda (anaconda3) as the primary dependency management system. You can install conda for free via https://www.anaconda.com/download.

#### 0.2 Install mamba (*strongly* recommended)
If you have conda but not mamba, I strongly recommend using mamba for setting up this project's environment. This environment is large and has *many* sub-dependencies, so it is likely that installing mamba first then running the setup instructions will take far less time than following the setup instructions using conda alone. If you already have mamba, you do not need to reinstall mamba.

```
# To install mamba
conda install -n base -c conda-forge mamba
```

### 1. Clone repo
Navigate to your target directory in terminal, then run:
```
git clone https://github.com/cielivb/NTPDVND
```

### 2. Download raw data files

To retrieve the raw data files, navigate to the project's data directory in terminal, then run the following commands to download the raw data into the data directory. On my local machine, the ~800 MB file took about 3 minutes to download, and the big one took about 13 minutes. You can also download these files manually from Zenodo, but I would recommend this only if you have a lot of patience and a Zenodo account. I could not successfully download the larger file manually on Windows 11.
```
aria2c -x 16 -s 16 -k 1M -o proofread_connections_783.feather https://zenodo.org/records/10676866/files/proofread_connections_783.feather
aria2c -x 16 -s 16 -k 1M -o flywire_synapses_783.feather https://zenodo.org/records/10676866/files/flywire_synapses_783.feather
```
To checksum the downloaded raw data:
```
# proofread_connections_783.feather MD5 checksum should be f48f972d262323a102aed49af1396b8a
Get-FileHash proofread_connections_783.feather -Algorithm MD5
# flywire_synapses_783.feather MD5 checksum should be f8f1b97c9d4b0ea9b4c8b287f6b99091
Get-FileHash flywire_synapses_783.feather -Algorithm MD5 
```

### 3. Set up environment

#### 1. Change directory to project root directory

#### 2. Create the conda environment
This installs Python 3.12, R 4.3, and required Python packages. This is environment contains large packages with many dependencies, so this may take a while. On my Windows 11 using mamba, this took ~5 minutes.
```
# If you use conda only:
conda env create -f environment.yml

# If you use mamba (MUCH faster than conda):
mamba env create -f environment.yml
```

#### 3. Activate and validate environment
```
conda activate nta3 # Activate the environment called nta3
python --version # Should show Python 3.12.x
R --version # Should show R 4.3.x
```

#### 4. Register environment as a Jupyter kernel
This makes the nta3 environment selectable as a kernel within Jupyter notebook.
```
python -m ipykernel install --user --name=nta3 --display-name "Python (nta3)"
```
<del>
#### 5. Restore packages from renv.lock
Start R inside your activated environment:
```
R
```
It will take a moment for R to start up. It is ready once > appears. Once R is running:
```
install.packages("renv") # You will be prompted to select a mirror - select the country closest to you
renv::restore() # Install packages specified in renv.lock
q() # Exit R session
```
</del>

### 4. Open Jupyter notebook

You should still be in your project root directory, and your environment should still be active. Now run:
```
jupyter notebook
```
This will take a few moments. A browser should automatically open showing your root directory. Next, click on the .ipynb file. Once open, select the Python (nta3) kernel in the top right corner to ensure the correct packages are used in your Jupyter Notebook session.


## Original Data Source
Full raw data metadata is available at the data source repo: [https://zenodo.org/records/10676866](https://zenodo.org/records/10676866)

### Files used
1. proofread_connections_783.feather - the primary dataset containing proofread neural connections
2. flywire_synapses_783.feather - supplies synapse coordinates, handy for 3D visualisation and weighting compositional analyses
