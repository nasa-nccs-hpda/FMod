
# FMod

Foundation Models based on the Nvidia Modulus Framework with MERRA2 training / fine tuning.

## Installation


#### OSX Local Development Installation
    * conda create -n fmod python=3.10 mamba -c conda-forge 
    * conda activate fmod
    * mamba install pytorch torchvision -c pytorch
    * mamba install -c dglteam dgl 
    * mamba install -c conda-forge s3fs tqdm pydantic ipython h5py h5netcdf matplotlib scipy netCDF4 ipympl jupyterlab ipykernel ipywidgets numpy xarray dask  pandas typing_extensions
    * pip install hydra-core --upgrade
    * pip install --no-deps nvidia-modulus nvidia-modulus-sym

#### Production Server Installation

 * Prerequesites:
   *  Must have mpicc in PATH  (e.g. */app/openmpi/platform/x86_64/rhel/8.9/4.1.2_gcc-12.1.0/bin/mpicc* )
   
 * Install Packages:

    >   * conda create -n fmod python=3.10 cuda-python -c nvidia 
    >   * conda activate fmod
    >   * pip install ninja
    >   * pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
    >   * pip install lightning-bolts 
    >   * pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" pytorch-extension
    >   * pip install nvidia-modulus[all] nvidia-modulus-sym
    >   * pip install netCDF4 h5py h5netcdf parameterized cartopy
    >   * cd torch-harmonics; pip install .

* Alternate pytorch-extension installation
  >  * cd /explore/nobackup/projects/ilab/software/pytorch-extension-0.2
  >  * python setup.py egg_info
  >  * python setup.py bdist_wheel 
  >  * python setup.py install
  >  * python setup.py clean


 * For graphcast:
    >   * pip install  dgl -f https://data.dgl.ai/wheels/cu121/repo.html
    >   * pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html


    >   * pip install wandb pydantic quadpy orthopy ndim gdown netCDF4 h5py h5netcdf mlflow torch-harmonics 

 * [GraphCast]: Install pymesh from source: https://pymesh.readthedocs.io/en/latest/installation.html
 * [SFNO]: Install apex from source: https://github.com/NVIDIA/apex












