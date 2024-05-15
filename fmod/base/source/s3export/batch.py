import xarray
import xarray as xa, math, os
from fmod.base.util.config import cfg
import pandas as pd
from datetime import  datetime, timedelta
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

def cut_tile( data_grid: np.ndarray, origin: Tuple[int,int] ):
	tile_size: int = cfg().task.tile_size
	return data_grid[origin[0]: origin[0] + tile_size, origin[1]: origin[1] + tile_size]

def cut_coord( cdata: np.ndarray, origin: int ) -> np.ndarray:
	tile_size: int = cfg().task.tile_size
	return cdata[ origin: origin + tile_size ]

def coords_filepath() -> str:
	return f"{cfg().platform.dataset_root}/xy_coords.nc"

def datelist( date_range: Tuple[datetime, datetime] ) -> pd.DatetimeIndex:
	return pd.date_range( date_range[0], date_range[1], freq=f"{cfg().task.hours_per_step}H" )

class S3ExportReader:

	def __init__(self, vres: srRes, **kwargs):
		self.vres: srRes = vres
		self.coords_dataset: xa.Dataset = xa.open_dataset( coords_filepath(), **kwargs)
		coords: Dict[str,str] = cfg().task.coords
		self.x: xa.DataArray = self.coords_dataset.data_vars[coords['x']]     # (y,x)
		self.y: xa.DataArray = self.coords_dataset.data_vars[coords['y']]     # (y,x)
		self.i: np.ndarray = self.coords_dataset.coords['i'].values
		self.j: np.ndarray = self.coords_dataset.coords['j'].values
		print(f" x coord shape = {self.x.shape} (y,x)")
		print(f" y coord shape = {self.y.shape} (y,x)")

	def open_datafile( self, varname: str, date: datetime ) -> np.ndarray:
		fpath = data_filepath( varname, date, self.vres )
		vardata: np.ndarray = np.load( fpath, allow_pickle=True, mmap_mode='r' )
		return vardata

	def load_channel( self, origin: Tuple[int,int], varname: str, date: datetime ) -> xa.DataArray:
		raw_data: np.ndarray = self.open_datafile( varname, date )                                        # (y,x)
		print( f"Raw data shape = {raw_data.shape} (y,x)")
		tile_data: np.ndarray = cut_tile( raw_data, origin )
		tcoords = dict( i=cut_coord( self.i, origin[1] ), j=cut_coord( self.j, origin[0] ) )
		xc: xa.DataArray = xa.DataArray( cut_tile( self.x.values, origin ), dims=['j','i'], coords=tcoords )
		yc: xa.DataArray = xa.DataArray( cut_tile( self.y.values, origin ), dims=['j','i'], coords=tcoords )
		print(f" xc shape {xc.shape} (y,x)")
		print(f" yc shape {yc.shape} (y,x)")
		print(f" tile_data shape {tile_data.shape} (y,x)")
		return xa.DataArray( tile_data, name=varname, dims=['y', 'x'], coords={'x': xc, 'y': yc} )

	def load_timeslice( self, origin: Tuple[int,int], varnames: List[str], date: datetime ) -> xarray.DataArray:
		arrays: List[xa.DataArray] = [ self.load_channel( origin, varname, date ) for varname in varnames ]
		return xa.concat(arrays, xa.DataArray(name='channel',data=np.array(varnames), dims=['cidx'], coords={'cidx': range(0,len(varnames))} ) )

	def load_temporal_batch( self, origin: Tuple[int,int], varnames: List[str], date_range: Tuple[datetime,datetime] ) -> xarray.DataArray:
		timeslices = [ self.load_timeslice(origin, varnames, date) for date in datelist( date_range ) ]

