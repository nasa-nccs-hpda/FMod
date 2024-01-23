
# FMod

Foundation Models based on the Nvidia Modulus Framework with MERRA2 training / fine tuning.

## Installation


#### OSX Local Development Installation
    * conda create -n fmod python=3.9 mamba -c conda-forge 
    * conda activate fmod
    * mamba install pytorch torchvision -c pytorch
    * mamba install -c dglteam dgl 
    * mamba install -c conda-forge s3fs tqdm pydantic ipython h5py h5netcdf matplotlib scipy netCDF4 ipympl jupyterlab ipykernel ipywidgets numpy xarray dask  pandas typing_extensions
    * pip install hydra-core --upgrade
    * pip install --no-deps nvidia-modulus nvidia-modulus-sym

#### Production Server Installation

 * Install Packages:
    >   * conda create -n fmod1 python=3.10 cudatoolkit-dev libcufile torchdata dgl -c conda-forge -c nvidia -c pytorch -c dglteam/label/cu121
    >   * conda activate fmod1
    >   * pip install --pre nvfuser-cu121 --extra-index-url https://pypi.nvidia.com
    >   * pip install nvidia-modulus[all] nvidia-modulus-sym
    >   * pip install wandb pydantic quadpy orthopy ndim gdown netCDF4 h5py h5netcdf mlflow torch-harmonics 

 * [GraphCast]: Install pymesh from source: https://pymesh.readthedocs.io/en/latest/installation.html
 * [SFNO]: Install apex from source: https://github.com/NVIDIA/apex












