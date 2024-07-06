import torch, logging, numpy as np
import hydra, os, time
from datetime import datetime
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping
from fmod.base.util.dates import date_range, cfg_date_range
import xarray as xa
from fmod.base.util.config import fmconfig, cfg
from fmod.base.util.logging import lgm, exception_handled, log_timing
from fmod.base.source.s3export.batch import S3ExportReader

hydra.initialize(version_base=None, config_path="../../config")
task="sres"
model="mscnn"
dataset="LLC4320"
scenario="s1"
fmconfig( task, model, dataset, scenario )

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
	torch.cuda.set_device(device.index)

location: Tuple[int,int] = (0,0)
start_date: datetime = datetime( 2012,1,12,1 )
end_date:   datetime = datetime( 2012,1,15,1 )
vres = srRes.High

reader = S3ExportReader( vres )
batch: xa.DataArray = reader.load_temporal_batch(location,(start_date,end_date))

print( "Read batch:")
print( f"* shape:     {batch.shape}")
print( f"* dims:      {batch.dims}")
print( f"* channel:   {batch.coords['channel'].values.tolist()}")
print( f"* batch[:4]: {batch.coords['batch'].values[:4]}\n\n")


