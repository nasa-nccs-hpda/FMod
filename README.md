
# FMod

Foundation Models based on the Nvidia Modulus Framework with MERRA2 training / fine tuning.

### Environment Creation

    > conda create -n fmod -c conda-forge 
    > conda activate fmod


#### OSX Local Development Installation

    > conda install pytorch torchvision -c pytorch
    > conda install -c dglteam dgl 
    > conda install -c conda-forge s3fs tqdm    ipython h5py h5netcdf matplotlib scipy netCDF4 ipympl jupyterlab ipykernel ipywidgets numpy xarray dask  pandas typing_extensions
    > pip install hydra-core --upgrade
    > pip install --no-deps nvidia-modulus nvidia-modulus-sym


#### Production Server Installation

    > pip install nvidia-modulus nvidia-modulus-sym
    > conda install cuda -c nvidia
    > conda install -c dglteam/label/cu121 dgl
    > pip install hydra-core --upgrade
    > pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda120
    > conda install -c conda-forge ipython h5py h5netcdf xarray netCDF4 ipympl jupyterlab ipykernel ipywidgets pandas typing_extensions


### Nvidia drivers installation

    > python3 -m pip install nvidia-pyindex
    > python3 -m pip install nvidia-cuda-runtime-cu12
    > python3 -m pip install nvidia-npp-cu123
    > python3 -m pip install nvidia-dali-cuda120 




