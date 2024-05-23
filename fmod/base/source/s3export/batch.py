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

	def __init__(self, task_config: DictConfig, tile_size: Dict[str, int], vres: srRes, **kwargs):
		SRDataLoader.__init__(self, task_config, vres )
		self.coords_dataset: xa.Dataset = xa.open_dataset( coords_filepath(), **kwargs)
#		self.xyc: Dict[str,xa.DataArray] = { c: self.coords_dataset.data_vars[ self.task.coords[c] ] for c in ['x','y'] }
#		self.ijc: Dict[str,np.ndarray]   = { c: self.coords_dataset.coords['i'].values.astype(np.int64) for c in ['i','j'] }
		self.tile_size: Dict[str,int] = tile_size
		self.varnames: Dict[str, str] = self.task.input_variables
		self.memmaps = Dict[str,np.memmap] = {}

	# def cut_coord(self, oindx: Dict[str,int], c: str) -> np.ndarray:
	# 	cdata: np.ndarray = self.ijc[c]
	# 	return cdata[origin[i2x(c)]: origin[i2x(c)] + self.tile_size[i2x(c)] ]

	def cut_tile( self, idx: int, data_grid: np.ndarray, origin: Dict[str,int] ):
		tile_bnds = [ origin['y'], origin['y'] + self.tile_size['y'], origin['x'], origin['x'] + self.tile_size['x'] ]
		if idx == 0: lgm().log( f"     ------------------>> cut_tile: origin={origin}, tile_bnds = {tile_bnds}")
		return data_grid[ tile_bnds[0]: tile_bnds[1], tile_bnds[2]: tile_bnds[3] ]

	# def cut_xy_coords(self, oindx: Dict[str,int] )-> Dict[str,xa.DataArray]:
	# 	tcoords: Dict[str,np.ndarray] = { c:  self.cut_coord( origin, c ) for idx, c in enumerate(['i','j']) }
	# #	xycoords: Dict[str,xa.DataArray] = { cv: xa.DataArray( self.cut_tile( self.xyc[cv].values, origin ), dims=['j','i'], coords=tcoords ) for cv in ['x','y'] }
	# #	xycoords: Dict[str, xa.DataArray] = {cv[0]: xa.DataArray(tcoords[cv[1]].astype(np.float32), dims=[cv[1]], coords=tcoords) for cv in [('x','i'), ('y','j')]}
	# 	xc = xa.DataArray(tcoords['i'].astype(np.float32), dims=['i'], coords=dict(i=tcoords['i']))
	# 	yc = xa.DataArray(tcoords['j'].astype(np.float32), dims=['j'], coords=dict(j=tcoords['j']))
	# 	return dict(x=xc, y=yc) #, **tcoords)

	def memmap_batch_data( self, date_range: Tuple[datetime,datetime] ):
		self.memmaps = {}
		for date in datelist(date_range):
			for vid in self.varnames.keys():
				self.memmaps[(vid,date)] = self.memmap_timeslice( vid, date )

	def memmap_timeslice(self, vid: str, date: datetime ) -> np.memmap:
		fpath = data_filepath(vid, date, self.vres)
		raw_data: np.memmap = np.load(fpath, allow_pickle=True, mmap_mode='r')
		return raw_data

	def load_channel( self, idx: int, origin: Dict[str,int], vid: Tuple[str,str], date: datetime ) -> xa.DataArray:
		raw_data: np.memmap = self.memmaps.get( (vid[0],date), self.memmap_tileslice(vid[0], date) )
		tile_data: np.ndarray = self.cut_tile( idx, raw_data, origin )
	#	tc: Dict[str,xa.DataArray] = self.cut_xy_coords(origin)
		if idx == 0: lgm().log( f" $$ load_channel: raw_data{raw_data.shape}, tile_data{tile_data.shape}, origin={origin}")
		result = xa.DataArray( tile_data, dims=['y', 'x'],  attrs=dict( fullname=vid[1] ) ) # coords=dict(**tc, **tc['x'].coords, **tc['y'].coords),
		return result.expand_dims( axis=0, dim=dict(channel=[vid[0]]) )

	def load_timeslice( self, idx: int, origin: Dict[str,int], date: datetime ) -> xa.DataArray:
		arrays: List[xa.DataArray] = [ self.load_channel( idx, origin, vid, date ) for vid in self.varnames.items() ]
		result = xa.concat( arrays, "channel" )
		result = result.expand_dims(axis=0, dim=dict(time=[np.datetime64(date)]))
		return result

	def load_temporal_batch( self, origin: Dict[str,int], date_range: Tuple[datetime,datetime] ) -> xa.DataArray:
		timeslices = [ self.load_timeslice( idx, origin,  date ) for idx, date in enumerate( datelist( date_range ) ) ]
		result = xa.concat(timeslices, "time")
		lgm().log( f" ** load-batch-{self.vres.value} [{date_range[0]}]:{result.dims}:{result.shape}, origin={origin}, tilesize = {self.tile_size}")
		return result

	def load_norm_data(self) -> Dict[str,xa.DataArray]:
		return {}

	def load_batch(self, origin: Dict[str,int], date_range: Tuple[datetime,datetime] ) -> xa.DataArray:
		darray: xa.DataArray = self.load_temporal_batch( origin, date_range )
		return darray

	def load_const_dataset(self, origin: Dict[str,int] )-> Optional[xa.DataArray]:
		return None





