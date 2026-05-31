# NTPDVND - Neurotransmitter Probability Distribution Variance Across Neuropils In Drosophila

## Replicate My Results!

**⚠️ DISCLAIMER:**: I have only tested this on Windows 11.

### Steps Summary
1. Clone repo
2. Download the raw data files into the project's data directory
3. Set up environment
4. Run `jupyter notebook`, then open either the exploratory notebook or performance testing notebook.

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

This project uses conda and renv to manage dependencies, and uses conda-forge as the python package source.

#### 1. Change directory to project root directory

#### 2. Enforce conda-forge strict priority (recommended)
This prevents implicit dependency checks against other package sources, which speeds up installation. You should only run this once. 
```
# If you use conda only:
conda config --add channels conda-forge
conda config --set channel_priority strict

# If you use mamba (faster):
mamba config --add channels conda-forge
mamba config --set channel_priority strict
```

#### 3. Create the conda environment
This installs Python 3.12, R 4.3, and required Python packages.
```
# If you use conda only:
conda env create -f environment.yml

# If you use mamba (faster):
mamba env create -f environment.yml
```

#### 4. Undo step 2 of environment set-up (optional)
This step is optional and only relevant if you ran step 2 to enforce conda-forge strict priority. If using mamba, replace 'conda' with 'mamba'.
```
# To reset priority rules:
conda config --set channel_priority flexible

# To remove conda-forge channel:
conda config --show channels # check what channels you currently have
conda config --remove channels conda-forge
conda config --show channels # conda-forge channel should be gone now
```

#### 4. Activate and validate environment
```
conda activate nta2 # Activate the environment called nta2
python --version # Should show Python 3.12.x
R --version # Should show R 4.3.x
```

#### 5. Register environment as a Jupyter kernel
This makes the nta2 environment selectable as a kernel within Jupyter notebook.
```
python -m ipykernel install --user --name=nta2 --display-name "Python (nta2)"
```

#### 6. Restore packages from renv.lock
Start R inside your activated environment:
```
R
```
It will take a moment for R to start up. Once R is running:
```
install.packages("renv") # Required for next command
renv::restore() # Install packages specified in renv.lock
q() # Exit R session
```

### 4. Open Jupyter notebook

You should still be in your project root directory, and your environment should still be active. Now run:
```
jupyter notebook
```
This will take a few moments. A browser should automatically open showing your root directory. Next, click on either .ipynb file (whichever you want to open). Once open, select the Python (nta2) kernel in the top right corner to ensure the correct packages are used in your Jupyter Notebook session.


## Original Data Source
Full raw data metadata is available at the data source repo: [https://zenodo.org/records/10676866](https://zenodo.org/records/10676866)

### Files used
1. proofread_connections_783.feather - the primary dataset containing proofread neural connections
2. flywire_synapses_783.feather - supplies synapse coordinates, handy for 3D visualisation and weighting compositional analyses



### Downloading required files

