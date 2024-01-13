
# FMod

Foundation Models based on the Nvidia Modulus Framework with MERRA2 training / fine tuning.

### Environment Creation

    > conda create -n fmod-gds python=3.8 cudatoolkit libcufile-static  -c conda-forge -c nvidia
    > conda activate fmod-gds


#### OSX Local Development Installation

    > mamba install pytorch torchvision -c pytorch
    > mamba install -c dglteam dgl 
    > mamba install -c conda-forge s3fs tqdm    ipython h5py h5netcdf matplotlib scipy netCDF4 ipympl jupyterlab ipykernel ipywidgets numpy xarray dask  pandas typing_extensions
    > pip install hydra-core --upgrade
    > pip install --no-deps nvidia-modulus nvidia-modulus-sym


#### Production Server Installation
    > pip install nvidia-modulus nvidia-modulus-sym
    > pip install quadpy orthopy ndim gdown netCDF4 h5py h5netcdf



 #   > mamba install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
  #   > mamba install -c conda-forge -c nvidia cudatoolkit  libcufile-static 
 # libnvjpeg libnpp libcufft

  #  > pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda120
 #    > pip install nvidia-modulus nvidia-modulus-sym
  #   > mamba install -c conda-forge ipython h5py h5netcdf xarray numpy netCDF4 ipympl jupyterlab ipykernel ipywidgets pandas typing_extensions
 #
 #    > pip install hydra-core --upgrade


 #    > conda install -c dglteam/label/cu121 dgl
  #   > pip install nvidia-modulus nvidia-modulus-sym
  #   > pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda120








