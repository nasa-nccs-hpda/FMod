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
		self.ijc: Dict[str,np.ndarray]   = { c: self.coords_dataset.coords['i'].values.astype(np.int64) for c in ['i','j'] }
		self.tile_size: Dict[str,int] = self.scale_coords( self.task.tile_size )
		self.varnames: Dict[str, str] = self.task.input_variables

	def scale_coords(self, c: Dict[str,int]) -> Dict[str,int]:
		if self.vres == srRes.Low:  return c
		else:                       return { k: v * self.scalefactor for k, v in c.items() }

	def cut_coord(self, oindx: Dict[str,int], c: str) -> np.ndarray:
		origin = self.scale_coords(oindx)
		cdata: np.ndarray = self.ijc[c]
		return cdata[origin[i2x(c)]: origin[i2x(c)] + self.tile_size[i2x(c)] ]

	def cut_tile( self, data_grid: np.ndarray, oindx: Dict[str,int] ):
		origin = self.scale_coords(oindx)
		return data_grid[ origin['y']: origin['y'] + self.tile_size['y'], origin['x']: origin['x'] + self.tile_size['x'] ]

	def cut_xy_coords(self, oindx: Dict[str,int] )-> Dict[str,xa.DataArray]:
		origin = self.scale_coords(oindx)
		tcoords: Dict[str,np.ndarray] = { c:  self.cut_coord( origin, c ) for idx, c in enumerate(['i','j']) }
	#	xycoords: Dict[str,xa.DataArray] = { cv: xa.DataArray( self.cut_tile( self.xyc[cv].values, origin ), dims=['j','i'], coords=tcoords ) for cv in ['x','y'] }
	#	xycoords: Dict[str, xa.DataArray] = {cv[0]: xa.DataArray(tcoords[cv[1]].astype(np.float32), dims=[cv[1]], coords=tcoords) for cv in [('x','i'), ('y','j')]}
		xc = xa.DataArray(tcoords['i'].astype(np.float32), dims=['i'], coords=dict(i=tcoords['i']))
		yc = xa.DataArray(tcoords['j'].astype(np.float32), dims=['j'], coords=dict(j=tcoords['j']))
		return dict(x=xc, y=yc) #, **tcoords)

	def load_channel( self, oindx: Dict[str,int], vid: Tuple[str,str], date: datetime ) -> xa.DataArray:
		origin = self.scale_coords(oindx)
		fpath = data_filepath(vid[0], date, self.vres)
		raw_data: np.memmap = np.load( fpath, allow_pickle=True, mmap_mode='r' )
		tile_data: np.ndarray = self.cut_tile( raw_data, origin )
		tc: Dict[str,xa.DataArray] = self.cut_xy_coords(origin)
		result = xa.DataArray( tile_data, dims=['j', 'i'], coords=dict(**tc, **tc['x'].coords, **tc['y'].coords), attrs=dict( fullname=vid[1] ) )
		return result.expand_dims( axis=0, dim=dict(channel=[vid[0]]) )

	def load_timeslice( self, oindx: Dict[str,int], date: datetime ) -> xa.DataArray:
		origin = self.scale_coords(oindx)
		arrays: List[xa.DataArray] = [ self.load_channel( origin, vid, date ) for vid in self.varnames.items() ]
		result = xa.concat( arrays, "channel" )
		return result.expand_dims(axis=0, dim=dict(time=[np.datetime64(date)]))

	def load_temporal_batch( self, oindx: Dict[str,int], date_range: Tuple[datetime,datetime] ) -> xa.DataArray:
		origin = self.scale_coords(oindx)
		timeslices = [ self.load_timeslice(origin,  date ) for date in datelist( date_range ) ]
		return xa.concat(timeslices, "time")

	def load_norm_data(self) -> Dict[str,xa.DataArray]:
		return {}

	def load_dataset(self, name: str, oindx: Dict[str,int], date_range: Tuple[datetime,datetime] ) -> xa.DataArray:
		origin = self.scale_coords(oindx)
		darray: xa.DataArray = self.load_temporal_batch( origin, date_range )
		return darray

	def load_const_dataset(self, origin: Dict[str,int] )-> Optional[xa.DataArray]:
		return None



