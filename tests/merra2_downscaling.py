import torch
import hydra, os, time
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping
from fmod.base.util.dates import date_list
from fmod.pipeline.downscale import Downscaler
from fmod.base.util.config import configure, cfg, cfg_date, cfg2args, pp
import xarray as xa
from fmod.pipeline.rescale import DataLoader
from datetime import date

hydra.initialize(version_base=None, config_path="../config")
configure('merra2-sr')
reference_date = date(1990, 6, 1)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(device.index)

data_loader = DataLoader()
lowres: xa.DataArray  = data_loader.get_channel_array( "low",  reference_date )
highres: xa.DataArray = data_loader.get_channel_array( "high", reference_date )

downscaler = Downscaler()
results: Dict[str,xa.DataArray] = downscaler.process( lowres, highres )
