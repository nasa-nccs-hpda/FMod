import torch, logging, numpy as np
import hydra, os, time
from datetime import datetime
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping
from fmod.base.util.dates import date_range, cfg_date_range
import xarray as xa
from fmod.base.util.config import fmconfig, cfg
from fmod.base.util.logging import lgm, exception_handled, log_timing
from fmod.base.source.s3export.batch import load_channel, srRes, S3ExportReader

hydra.initialize(version_base=None, config_path="../config")
task="sres"
model="mscnn"
dataset="s3export"
scenario="s1"
fmconfig( task, model, dataset, scenario )

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
	torch.cuda.set_device(device.index)

origin: Tuple[int,int] = (0,0)
varnames: List[str] = [ 'sst']
date: datetime = datetime( 2012,1,12,15 )
vres = srRes.High

reader = S3ExportReader( vres )
timeslice: xa.DataArray = reader.load_timeslice(origin,varnames,date)

print(timeslice.shape)
print(timeslice.dims)

