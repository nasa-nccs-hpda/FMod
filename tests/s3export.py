import torch, logging
import hydra, os, time
from datetime import datetime
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping
from fmod.base.util.dates import date_range, cfg_date_range
from fmod.base.util.config import fmconfig, cfg
from fmod.base.util.logging import lgm, exception_handled, log_timing
from fmod.base.source.s3export.batch import load_channel, srRes

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
date: datetime = datetime( )
vres = srRes.High

batch = load_channel( origin, varnames[0], date, vres )

