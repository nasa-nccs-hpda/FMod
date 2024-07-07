import xarray as xa, math, os
from fmod.base.util.config import cfg, dateindex
import pandas as pd
from datetime import  datetime, timedelta
from omegaconf import DictConfig, OmegaConf
from fmod.base.util.dates import drepr, date_list
from nvidia.dali import fn
from enum import Enum
from fmod.base.io.loader import TSet, srRes
from glob import glob
from .raw import SWOTRawDataLoader
from typing import Any, Mapping, Sequence, Tuple, Union, List, Dict, Literal, Optional
from fmod.base.util.ops import format_timedeltas
from fmod.base.io.loader import data_suffix, path_suffix
from fmod.base.util.logging import lgm, exception_handled, log_timing
from fmod.base.source.loader.batch import SRDataLoader
from fmod.base.util.config import start_date
import numpy as np
#
# S = 'x'
# CoordIdx = Union[ Dict[str,int], Tuple[int,int] ]
#
# def cTup2Dict( c: CoordIdx ) -> Dict[str,int]:
# 	if type(c) is tuple: c = dict(x=c[0], y=c[1])
# 	return c
#
# def dstr(date: datetime) -> str:
# 	return '{:04}{:02}{:02}{:02}'.format( date.year, date.month, date.day, date.hour )
#
#
# def i2x( c: str ) -> str:
# 	if c == "i": return "x"
# 	if c == "j": return "y"
#
# def get_version(task_config: DictConfig) -> int:
# 	toks = task_config.dataset.split('-')
# 	tstr = "v0" if (len(toks) == 1) else toks[-1]
# 	assert tstr[0] == "v", f"Version str must start with 'v': '{tstr}'"
# 	return int(tstr[1:])
#
# def datelist( date_range: Tuple[datetime, datetime] ) -> pd.DatetimeIndex:
# 	dlist =  pd.date_range( date_range[0], date_range[1], freq=f"{cfg().task.hours_per_step}h", inclusive="left" )
# 	# print( f" ^^^^^^ datelist[ {date_range[0].strftime('%H:%d/%m/%Y')} -> {date_range[1].strftime('%H:%d/%m/%Y')} ]: {dlist.size} dates, start_date = {cfg().task['start_date']}" )
# 	return dlist
#
# def scale( varname: str, batch_data: np.ndarray ) -> np.ndarray:
# 	ranges: Dict[str,Dict[str,float]] = cfg().task.variable_ranges
# 	vrange: Dict[str,float] = ranges[varname]
# 	return (batch_data - vrange['min']) / (vrange['max'] - vrange['min'])
#
# def tcoord( ** kwargs ) :
# 	dindx = kwargs.get('index',-1)
# 	date: Optional[datetime] = kwargs.get('date',None)
# 	return dindx if (date is None) else np.datetime64(date)

def filepath(ftype: str ) -> str:
	return f"{cfg().dataset.dataset_root}/{cfg().dataset.dataset_files[ftype]}"

import numpy as np
import matplotlib.pyplot as plt
# import utils
CoordIdx = Union[ Dict[str,int], Tuple[int,int] ]

class SWOTDataLoader(SRDataLoader):

	def __init__(self, task_config: DictConfig, vres: srRes,  **kwargs):
		SRDataLoader.__init__(self, task_config, vres)
		self.loader = SWOTRawDataLoader(task_config,  **kwargs)
		self.shape = None

	def load_tile_batch(self, tile_index: int, time_index: int, tset: TSet ) -> xa.DataArray:
		tile_batch: xa.DataArray = self.loader.load_batch( tile_index, time_index, tset )
		return tile_batch

	def get_batch_time_indices(self):
		return self.loader.get_batch_time_indices()






