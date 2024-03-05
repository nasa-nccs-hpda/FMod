import torch, logging
import hydra, os, time
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping
from fmod.base.util.dates import date_list
from fmod.base.util.config import configure, cfg, cfg_date, cfg2args, pp
from fmod.pipeline.merra2 import MERRA2Dataset
from fmod.base.util.logging import lgm, exception_handled, log_timing

hydra.initialize(version_base=None, config_path="../config")
configure('merra2-sr')
cfg().task.device = "cpu"
lgm().set_level( logging.DEBUG )

def nnan(varray: torch.Tensor) -> int: return torch.isnan(varray).sum().item()
def pctnan(varray: torch.Tensor) -> str: return f"{nnan(varray)*100.0/torch.numel(varray):.2f}%"

# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(device.index)

dataset = MERRA2Dataset( train_dates=date_list( cfg_date('task'), cfg().task.max_steps ), vres="low" )
data_iter = iter(dataset)

for inp, tar in data_iter:
	print(f" ** inp shape={inp.shape}, pct-nan= {pctnan(inp)}")
	print(f" ** tar shape={tar.shape}, pct-nan= {pctnan(tar)}")
	break


