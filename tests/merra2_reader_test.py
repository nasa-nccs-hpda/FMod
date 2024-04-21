import torch, logging
import hydra, os, time
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping
from fmod.base.util.dates import date_list
from fmod.base.util.config import configure, cfg, start_date,  cfg2args, pp
from fmod.pipeline.merra2 import MERRA2Dataset
from fmod.base.util.logging import lgm, exception_handled, log_timing
from fmod.base.util.ops import pctnan

hydra.initialize(version_base=None, config_path="../config")
configure('merra2-sr')
cfg().task.device = "cpu"
lgm().set_level( logging.DEBUG )

dataset = MERRA2Dataset( train_dates=date_list( start_date( cfg().task ), cfg().task.max_steps ), vres="low" )
data_iter = iter(dataset)

for inp, tar in data_iter:
	print(f" ** inp shape={inp.shape}, pct-nan= {pctnan(inp)}")
	print(f" ** tar shape={tar.shape}, pct-nan= {pctnan(tar)}")
	break


