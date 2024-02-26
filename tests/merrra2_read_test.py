import torch
import hydra, os, time
from fmod.base.source.merra2.model import load_dataset
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping
from fmod.base.util.dates import date_list
from fmod.pipeline.downscale import Downscaler
from fmod.base.util.config import configure, cfg, cfg_date, cfg2args, pp
import xarray as xa
from fmod.pipeline.rescale import DataLoader
from datetime import date
from xarray.core.types import InterpOptions, Interp1dOptions

hydra.initialize(version_base=None, config_path="../config")
configure('merra2-sr')
reference_date = date(1992,1,3 )
vres = "high"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(device.index)

dset: xa.Dataset = load_dataset( vres, reference_date )

print(f"\n -------------------- Encodings -------------------- ")
for vid in dset.data_vars.keys():
	dvar: xa.DataArray = dset.data_vars[vid]
	print( f"*** {vid:<30} {str(dvar.dims):<35} shape={str(dvar.shape):<25} chunks={dvar.encoding['preferred_chunks']}")



