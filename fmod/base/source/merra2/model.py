import xarray as xa, pandas as pd
import os, math, numpy as np, shutil
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping, Literal
from fmod.pipeline.stats import StatsAccumulator
from fmod.base.util.dates import drepr, date_list
from fmod.base.util.config import get_data_indices, get_roi
from datetime import date
from fmod.base.util.logging import lgm, exception_handled, log_timing
from fmod.base.util.config import cfg
from fmod.base.source.merra2.batch import get_steps_per_day
from .batch import rename_vars, get_days_per_batch, get_target_steps, VarType, BatchType, cache_filepath, stats_filepath, rename_coords, subset_datavars
from fmod.base.io.loader import ncFormat
from fmod.base.util.ops import nnan, pctnan, remove_filepath

_SEC_PER_HOUR = 3600
_HOUR_PER_DAY = 24
SEC_PER_DAY = _SEC_PER_HOUR * _HOUR_PER_DAY
_AVG_DAY_PER_YEAR = 365.24219
AVG_SEC_PER_YEAR = SEC_PER_DAY * _AVG_DAY_PER_YEAR

def d2xa( dvals: Dict[str,float] ) -> xa.Dataset:
	return xa.Dataset( {vn: xa.DataArray( np.array(dval) ) for vn, dval in dvals.items()} )

def rcoords(dset: xa.Dataset):
	c = dset.coords
	return '[' + ','.join( [ f"{k}:{c[k].size}" for k in c.keys()] ) + ']'

def bounds(dset: xa.Dataset):
	c = dset.coords
	return '[' + ','.join( [ f"{k}[{c[k].size}]:[{c[k][0]:.2f},{c[k][-1]:.2f}:{c[k][1]-c[k][0]:.2f}]" for k in ['x','y']] ) + ']'

def index_of_cval(  data: Union[xa.Dataset,xa.DataArray], dim:str, cval: float)-> int:
	coord: np.ndarray = data.coords[dim].values
	cindx: np.ndarray = np.where(coord==cval)[0]
	return cindx.tolist()[0] if cindx.size else -1

def clear_const_file():
	for vres in ["high", "low"]:
		const_filepath = cache_filepath(VarType.Constant,vres=vres)
		remove_filepath(const_filepath)

def get_index_roi(dataset: xa.Dataset, vres: str) -> Optional[Dict[str,slice]]:
	roi: Optional[Dict[str, List[float]]] = cfg().task.get('roi')
	if roi is None: return None
	cbounds: Dict = {}
	for dim in ['x', 'y']:
		croi: List[float] = roi[dim]
		coord: List[float] = dataset.coords[dim].values.tolist()
		iroi: int =  index_of_cval( dataset, dim, croi[0] )
		crisize = round( (croi[1]-croi[0]) / (coord[1] - coord[0] ) )
		cbounds[dim] = slice( iroi, iroi + crisize )
	return cbounds

def furbish( dset: xa.Dataset ) -> xa.Dataset:
	dvars: Dict = { vname: dvar.squeeze() for vname, dvar in dset.data_vars.items() }
	coords: Dict = { cname: cval for cname, cval in dset.coords.items() }
	attrs: Dict = { aname: aval for aname, aval in dset.attrs }
	attrs['datetime'] = coords.pop('datetime', attrs.get('datetime',None) )
	return xa.Dataset( dvars, coords, attrs )

def merge_batch1( slices: List[xa.Dataset], constants: xa.Dataset ) -> xa.Dataset:
	constant_vars: List[str] = cfg().task.get('constants',[])
	cvars = [vname for vname, vdata in slices[0].data_vars.items() if "time" not in vdata.dims]
	vslices = [ furbish(vslice) for vslice in slices ]
	# lgm().log(f" ----- merge_batch ----- ")
	# for vslice in vslices:
	# 	if 'datetime' in vslice.coords:
	# 		lgm().log( f" **> slice coords: {list(vslice.coords.keys())} ----- ")
	# 		for vname, dvar in vslice.data_vars.items():
	# 			lgm().log( ' >>>>> {:30s} {:35s} {:25s} '.format(vname,str(dvar.dims),str(dvar.shape)) )
	dynamics: xa.Dataset = xa.concat( vslices, dim="time", coords = "minimal" )
	dynamics = dynamics.drop_vars(cvars)
	sample: xa.Dataset = vslices[0].drop_dims( 'time', errors='ignore' )
	for vname, dvar in sample.data_vars.items():
		if vname not in dynamics.data_vars.keys():
			constants[vname] = dvar
		elif (vname in constant_vars) and ("time" in dvar.dims):
			dvar = dvar.mean(dim="time", skipna=True, keep_attrs=True)
			constants[vname] = dvar
	dynamics = dynamics.drop_vars(constant_vars, errors='ignore')
	return xa.merge( [dynamics, constants], compat='override' )

def merge_batch( slices: List[xa.Dataset], constants: xa.Dataset ) -> xa.Dataset:
	dynamics: xa.Dataset = xa.concat( slices, dim="time", coords = "minimal" )
	return xa.merge( [dynamics, constants], compat='override' )

def get_predef_norm_data() -> Dict[str, xa.Dataset]:
	sndef = { sn:sn for sn in StatsAccumulator.statnames }
	snames: Dict[str, str] = cfg().task.get('statnames',sndef)
	dstd = dict(year_progress=0.0247, year_progress_sin=0.003, year_progress_cos=0.003, day_progress=0.433, day_progress_sin=1.0, day_progress_cos=1.0)
	vmean = dict(year_progress=0.5, year_progress_sin=0.0, year_progress_cos=0.0, day_progress=0.5, day_progress_sin=0.0, day_progress_cos=0.0)
	vstd = dict(year_progress=0.29, year_progress_sin=0.707, year_progress_cos=0.707, day_progress=0.29, day_progress_sin=0.707, day_progress_cos=0.707)
	pdstats = dict(std_diff=d2xa(dstd), mean=d2xa(vmean), std=d2xa(vstd))
	return {snames[sname]: pdstats[sname] for sname in sndef.keys()}

@log_timing
def load_stats(statname: str, **kwargs) -> xa.Dataset:
	version = cfg().task['dataset_version']
	filepath = stats_filepath(version, statname)
	varstats: xa.Dataset = xa.open_dataset(filepath, **kwargs)
	return rename_vars( varstats )

@log_timing
def load_norm_data() -> Dict[str, xa.Dataset]:
	sndef = { sn:sn for sn in StatsAccumulator.statnames }
	snames: Dict[str, str] = cfg().task.get('statnames',sndef)
	return {} if snames is None else {snames[sname]: load_stats(sname) for sname in sndef.keys()}

@log_timing
def load_merra2_norm_data() -> Dict[str, xa.Dataset]:
	predef_norm_data: Dict[str, xa.Dataset] = get_predef_norm_data()
	m2_norm_data: Dict[str, xa.Dataset] = load_norm_data()
	lgm().log( f"predef_norm_data: {list(predef_norm_data.keys())}")
	lgm().log( f"m2_norm_data: {list(m2_norm_data.keys())}")
	return {nnorm: xa.merge([predef_norm_data[nnorm], m2_norm_data[nnorm]]) for nnorm in m2_norm_data.keys()}

def access_data_subset( filepath, vres: str ) -> xa.Dataset:
	levels: Optional[ List[float] ] = cfg().task.get('levels')
	dataset: xa.Dataset = subset_datavars( xa.open_dataset(filepath, engine='netcdf4') )
	if (levels is not None) and ('z' in dataset.coords):
		dataset = dataset.sel(z=levels, method="nearest")
	origin: Dict[str,float] = cfg().task.origin
	tile_size: Dict[str,int] = cfg().task.tile_size
	oindx: Dict[str,int] = get_data_indices(dataset, origin)
	if vres == "high":
		upscale_factor = math.prod( cfg().model.upscale_factors )
		tile_size = { dim: ts*upscale_factor for dim, ts in tile_size.items() }
	iroi = { dim: slice(oindx[dim], oindx[dim]+ts) for dim, ts in tile_size.items() }
	dataset = dataset.isel( **iroi )
	lgm().log( f"\n %% data_subset[{vres}]-> iroi: {iroi}, dataset roi: {get_roi(dataset.coords)}")
	return rename_coords(dataset)

def load_dataset(  d: date, vres: str="high" ) -> xa.Dataset:
	filepath =  cache_filepath( VarType.Dynamic, d, vres )
	result: xa.Dataset = access_data_subset( filepath, vres )
	print( f" * load_dataset[{vres}]({d}) {get_roi(result.coords)} {filepath}")
	return result

def load_const_dataset( vres: str = "high" ) -> xa.Dataset:
	filepath =  cache_filepath(VarType.Constant, vres=vres )
	return access_data_subset( filepath, vres )

class FMBatch:

	def __init__(self, btype: BatchType, **kwargs):
		self.format = ncFormat( cfg().task.get('nc_format', 'standard') )
		self.type: BatchType = btype
		self.vres = kwargs.get('vres', "high" )
		self.days_per_batch = get_days_per_batch(btype)
		self.target_steps = get_target_steps(btype)
		self.batch_steps: int = cfg().task.nsteps_input + len(self.target_steps)
		self.constants: xa.Dataset = load_const_dataset( **kwargs )
		self.norm_data: Dict[str, xa.Dataset] = load_merra2_norm_data()
		self.current_batch: xa.Dataset = None

	def load(self, d: date, **kwargs):
		bdays = date_list(d, self.days_per_batch)
		time_slices: List[xa.Dataset] = [ load_dataset(d, self.vres) for d in bdays ]
		self.current_batch: xa.Dataset = merge_batch(time_slices, self.constants)

	def get_train_data(self,  day_offset: int ) -> xa.Dataset:
		return self.current_batch.isel( time=slice(day_offset, day_offset+self.batch_steps) )

	def get_time_slice(self,  day_offset: int) -> xa.Dataset:
		return self.current_batch.isel( time=day_offset )

	@classmethod
	def to_feature_array( cls, data_batch: xa.Dataset) -> xa.DataArray:
		features = xa.DataArray(data=list(data_batch.data_vars.keys()), name="features")
		result = xa.concat( list(data_batch.data_vars.values()), dim=features )
		result = result.transpose(..., "features")
		return result

class SRBatch:

	def __init__(self, **kwargs):
		self.vres = kwargs.get('vres', "high" )
		self.current_batch: xa.Dataset = None
		self.current_date = None
		self.days_per_batch = cfg().task.batch_ndays
		self.batch_steps: int = self.days_per_batch * get_steps_per_day()
		self._constants: Optional[xa.Dataset] = None
		self.norm_data: Dict[str, xa.Dataset] = load_merra2_norm_data()

	@property
	def constants(self)-> xa.Dataset:
		if self._constants is None:
			self._constants: xa.Dataset = load_const_dataset(self.vres)
		return self._constants

	def merge_batch(self, slices: List[xa.Dataset]) -> xa.Dataset:
		dynamics: xa.Dataset = xa.concat(slices, dim="time", coords="minimal")
		merged: xa.Dataset =  xa.merge([dynamics, self.constants], compat='override')
		return merged

	def load_batch(self, d: date, **kwargs) -> xa.Dataset:
		dates = date_list(d, self.days_per_batch)
		dsets = [ load_dataset(day, self.vres) for day in dates ]
		print( f"Concat {len(dates)} daily datasets")
		dset = xa.concat(dsets, dim="time", coords="minimal")
		return dset

	def load(self, d: date) -> xa.Dataset:
		if self.current_date != d:
			self.current_batch = self.load_batch(d)
			self.current_date = d
			lgm().log( f"\n -----> load_batch[{d}]-> {rcoords(self.current_batch)}\n", display=True )

		return self.current_batch












