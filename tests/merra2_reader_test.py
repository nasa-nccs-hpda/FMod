import logging, torch
from fmod.base.util.logging import lgm, exception_handled, log_timing
from fmod.base.util.ops import nnan, pctnan, remove_filepath
import xarray as xa
import hydra, os, time
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping
from fmod.base.util.config import configure, cfg, start_date,  cfg2args, get_roi, get_data_coords
from fmod.pipeline.ncbatch import ncBatchDataset

hydra.initialize(version_base=None, config_path="../config")
configure('merra2-srdn-s1')
# lgm().set_level( logging.DEBUG )

load_state = False
best_model = False
save_state = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(device.index)

input_dataset   = ncBatchDataset( cfg().task, vres="low",  load_inputs=True,  load_base=False, load_targets=False )
sample_batch: xa.DataArray  = input_dataset.get_batch( input_dataset.train_dates[0] )['input']
print( f" @@@ sample_batch{sample_batch.dims}: shape={sample_batch.shape}, pctnan={pctnan(sample_batch.values)}" )

print( f"  ** cfg origin = {cfg().task['origin']}" )
data_origin = get_data_coords( sample_batch, cfg().task['origin'] )
print( f"  ** data origin = {data_origin}"  )
cfg().task['origin'] = data_origin
print( f"  ** updated cfg origin = {cfg().task['origin']}" )

print( f"  ** lowres roi = {get_roi(sample_batch.coords)}" )

target_dataset   = ncBatchDataset( cfg().task, vres="high",  load_inputs=False,  load_base=False, load_targets=True )
target_batch: xa.DataArray  = input_dataset.get_batch( input_dataset.train_dates[0] )['target']
print( f" @@@ sample_batch{target_batch.dims}: shape={target_batch.shape}, pctnan={pctnan(target_batch.values)}" )

print( f"  ** highres roi = {get_roi(target_batch.coords)}" )



