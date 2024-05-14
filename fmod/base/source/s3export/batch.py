import xarray as xa, math, os
from fmod.base.util.config import cfg
from datetime import  datetime
from fmod.base.util.dates import drepr, date_list
from nvidia.dali import fn
from enum import Enum
from typing import Any, Mapping, Sequence, Tuple, Union, List, Dict, Literal
from fmod.base.util.ops import format_timedeltas, fmbdir
from fmod.base.io.loader import data_suffix, path_suffix
from fmod.base.util.logging import lgm, exception_handled, log_timing
import numpy as np

class srRes(Enum):
	Low = 'lr'
	High = 'hr'

def dstr(date: datetime) -> str:
	return '{:04}{:02}{:02}{:02}'.format( date.year, date.month, date.day, date.hour )

def data_filepath( varname: str, date: datetime, vres: srRes ) -> str:
	root = cfg().platform.dataset_root
	subpath = cfg().platform.dataset_files.format( res=vres.value, varname=varname, date=dstr(date) )
	fpath = f"{root}/{subpath}"
	return fpath

def open_datafile( varname: str, date: datetime, vres: srRes ) -> np.ndarray:
	fpath = data_filepath( varname, date, vres )
	vardata: np.ndarray = np.load( fpath, allow_pickle=True, mmap_mode='r' )
	return vardata

def load_channel( origin: Tuple[int,int], varname: str, date: datetime, vres: srRes ) -> np.ndarray:
	tile_size: int = cfg().task.tile_size
	raw_data: np.ndarray = open_datafile( varname, date, vres )
	print( raw_data.shape )
	return raw_data