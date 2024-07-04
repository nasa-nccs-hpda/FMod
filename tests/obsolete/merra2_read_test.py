import torch
import hydra, os, time
from fmod.base.source.merra2.model import load_dataset
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping
from fmod.base.source.merra2.model import pctnan
from fmod.base.util.config import cfg, start_date,  cfg2args, pp
import xarray as xa
from datetime import date

hydra.initialize(version_base=None, config_path="../../config")
configure('merra2-sr')
reference_date = date(1990,1,1 )
vres = "high"

dset: xa.Dataset = load_dataset( reference_date, vres )

print(f"\n -------------------- Encodings: D=dims, S=shape, C=chunks, PN=%nan -------------------- ")
for vid in dset.data_vars.keys():
	dvar: xa.DataArray = dset.data_vars[vid]
	print( f"*** {vid:<30} D{str(dvar.dims):<35} S{str(dvar.shape):<22} C{dvar.encoding.get('preferred_chunks')}, PN: {pctnan(dvar)}")



