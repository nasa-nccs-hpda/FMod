import torch, logging
import hydra, os, time
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping
from fmod.base.util.dates import date_range, cfg_date_range
from fmod.base.util.config import cfg, start_date,  cfg2args, pp
from fmod.pipeline.ncbatch import ncBatchDataset
from fmod.base.util.logging import lgm, exception_handled, log_timing

hydra.initialize(version_base=None, config_path="../config")
configure('merra2-unet-s1')
lgm().set_level( logging.DEBUG )

# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
	torch.cuda.set_device(device.index)

input_dataset  = ncBatchDataset( cfg().task, load_inputs=True, load_base=False, load_targets=True )
results = next(iter(input_dataset))
sample_input = results['input']
print( f"sample_input{sample_input.dims}: {sample_input.shape} {list(sample_input.coords.keys())}" )
for k,c in sample_input.coords.items():
	print( f"{k}{c.shape}: {c[0]:.2f} -> {c[-1]:.2f}")
	break


