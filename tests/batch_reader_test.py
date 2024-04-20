import torch, logging
import hydra, os, time
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping
from fmod.base.util.dates import date_range, cfg_date_range
from fmod.base.util.config import configure, cfg, cfg_date, cfg2args, pp
from fmod.pipeline.ncbatch import ncBatchDataset
from fmod.base.util.logging import lgm, exception_handled, log_timing
from fmod.base.util.ops import pctnan

hydra.initialize(version_base=None, config_path="../config")
configure('merra2-srdn-s1')
cfg().task.device = "cpu"
lgm().set_level( logging.DEBUG )

dataset = ncBatchDataset( cfg().task, vres="low" )
data_iter = iter(dataset)

for inp, tar in data_iter:
	print(f" ** inp shape={inp.shape}, pct-nan= {pctnan(inp)}")
	print(f" ** tar shape={tar.shape}, pct-nan= {pctnan(tar)}")
	break


