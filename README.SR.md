
# FMod

Super Resolution Test and Development Framework.

## Conda environment

    >   * conda create -n sres mamba
    >   * conda activate sres
    >   * mamba install pytorch torchvision torchaudio pytorch-cuda cuda-python -c pytorch -c nvidia
    >   * pip install nvidia-dali-cuda120 
    >   * mamba install scipy ipykernel xarray netCDF4 ipympl 
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










