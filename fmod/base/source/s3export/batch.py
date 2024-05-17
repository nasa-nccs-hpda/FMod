import xarray as xa, math, os
from fmod.base.util.config import cfg
import pandas as pd
from datetime import  datetime, timedelta
from omegaconf import DictConfig, OmegaConf
from fmod.base.util.dates import drepr, date_list
from nvidia.dali import fn
from enum import Enum
from typing import Any, Mapping, Sequence, Tuple, Union, List, Dict, Literal, Optional
from fmod.base.util.ops import format_timedeltas, fmbdir
from fmod.base.io.loader import data_suffix, path_suffix
from fmod.base.util.logging import lgm, exception_handled, log_timing
from fmod.base.source.loader import srRes
from fmod.base.source.loader import SRDataLoader
import numpy as np

S = 'x'

def dstr(date: datetime) -> str:
	return '{:04}{:02}{:02}{:02}'.format( date.year, date.month, date.day, date.hour )

def data_filepath( varname: str, date: datetime, vres: srRes ) -> str:
	root: str = cfg().platform.dataset_root
	usf: int = math.prod( cfg().model.downscale_factors )
	subpath: str = cfg().platform.dataset_files[vres.value].format( res=vres.value, varname=varname, date=dstr(date), usf=usf )
	fpath = f"{root}/{subpath}"
	return fpath

def coords_filepath() -> str:
	return f"{cfg().platform.dataset_root}/xy_coords.nc"

def i2x( c: str ) -> str:
	if c == "i": return "x"
	if c == "j": return "y"

def datelist( date_range: Tuple[datetime, datetime] ) -> pd.DatetimeIndex:
	return pd.date_range( date_range[0], date_range[1], freq=f"{cfg().task.hours_per_step}h" )

class S3ExportDataLoader(SRDataLoader):

	def __init__(self, task_config: DictConfig, vres: str, **kwargs):
		SRDataLoader.__init__(self, task_config, vres )
		self.coords_dataset: xa.Dataset = xa.open_dataset( coords_filepath(), **kwargs)
		self.xyc: Dict[str,xa.DataArray] = { c: self.coords_dataset.data_vars[ self.task.coords[c] ] for c in ['x','y'] }
		self.ijc: Dict[str,np.ndarray]   = { c: self.coords_dataset.coords['i'].values for c in ['i','j'] }
		self.tile_size: Dict[str,int] = self.task.tile_size
		self.varnames: Dict[str, str] = self.task.input_variables

	def cut_coord(self, cdata: np.ndarray, origin: int) -> np.ndarray:
		tile_size: int = self.task.tile_size
		return cdata[origin: origin + tile_size]

	def cut_tile( self, data_grid: np.ndarray, origin: Dict[str,int] ):
		return data_grid[ origin['y']: origin['y'] + self.tile_size['y'], origin['x']: origin['x'] + self.tile_size['x'] ]

	def cut_xy_coords(self, origin: Dict[str,int] )-> Dict[str,xa.DataArray]:
		tcoords: Dict[str,np.ndarray] = { c:  self.cut_coord( self.ijc[c], origin[i2x(c)] ) for idx, c in enumerate(['i','j']) }
		xycoords: Dict[str,xa.DataArray] = { cv: xa.DataArray( self.cut_tile( self.xyc[cv].values, origin ), dims=['j','i'], coords=tcoords ) for cv in ['x','y'] }
		return xycoords

	def load_channel( self, origin: Dict[str,int], vid: Tuple[str,str], date: datetime ) -> xa.DataArray:
		fpath = data_filepath(vid[0], date, self.vres)
		raw_data: np.memmap = np.load( fpath, allow_pickle=True, mmap_mode='r' )
		tile_data: np.ndarray = self.cut_tile( raw_data, origin )
		tc: Dict[str,xa.DataArray] = self.cut_xy_coords(origin)
		result = xa.DataArray( tile_data, dims=['j', 'i'], coords=dict(**tc, **tc['x'].coords) )
		return result.expand_dims( axis=0, dim=dict(channel=[vid[1]]) )

	def load_timeslice( self, origin: Dict[str,int], date: datetime ) -> xa.DataArray:
		arrays: List[xa.DataArray] = [ self.load_channel( origin, vid, date ) for vid in self.varnames.items() ]
		result = xa.concat( arrays, "channel" )
		return result.expand_dims(axis=0, dim=dict(batch=[np.datetime64(date)]))

	def load_temporal_batch( self, origin: Dict[str,int], date_range: Tuple[datetime,datetime] ) -> xa.DataArray:
		timeslices = [ self.load_timeslice(origin,  date ) for date in datelist( date_range ) ]
		return xa.concat(timeslices, "batch")

	def load_norm_data(self) -> Dict[str,xa.DataArray]:
		return {}

	def load_dataset(self, origin: Dict[str,int], date_range: Tuple[datetime,datetime] ) -> xa.DataArray:
		return  self.load_temporal_batch( origin, date_range )

	def load_const_dataset(self)-> Optional[xa.DataArray]:
		return None



