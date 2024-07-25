
# FMod

Super Resolution Test and Development Framework.

## Conda environment

    >   * conda create -n sres mamba python=3.11
    >   * conda activate sres
    >   * mamba install -c conda-forge scipy xarray netCDF4  ipywidgets=7.8 jupyterlab=4.0 jupyterlab_widgets ipykernel=6.29 ipympl=0.9 ipython=8.26
    >   * mamba install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda cuda-python
    >   * pip install parse nvidia-dali-cuda120
    >   * pip install hydra-core --upgrade
    >   * ipython kernel install --user --name=sres

## Installation

    > git clone https://github.com/nasa-nccs-hpda/FMod.git
    > cd FMod/notebooks/
    > ln -s ../fmod ./fmod
    > cd ../scripts
    > ln -s ../fmod ./fmod

## Inference

    Run the jupyter notebook: FMod/notebooks/plot_results.ipynb

    This notebook executes the inference engine and displays the result 
    whenever the time or subtile indices change (via sliders at the bottom).










