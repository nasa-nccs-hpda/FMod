
# FMod

Foundation Models based on the Nvidia Modulus Framework with MERRA2 training / fine tuning.

#### OSX Local Development Installation

    > conda create -n fmod python=3.8 mamba -c conda-forge 
    > conda activate fmod
    > mamba install pytorch torchvision -c pytorch
    > mamba install -c dglteam dgl 
    > mamba install -c conda-forge s3fs tqdm pydantic ipython h5py h5netcdf matplotlib scipy netCDF4 ipympl jupyterlab ipykernel ipywidgets numpy xarray dask  pandas typing_extensions
    > pip install hydra-core --upgrade
    > pip install --no-deps nvidia-modulus nvidia-modulus-sym


#### Production Server Installation

    > conda create -n fmod-gds python=3.8 cudatoolkit libcufile dgl -c conda-forge -c nvidia -c dglteam/label/cu121
    > conda activate fmod-gds
    > pip install nvidia-modulus nvidia-modulus-sym
    > pip install wandb pydantic quadpy orthopy ndim gdown netCDF4 h5py h5netcdf mlflow










