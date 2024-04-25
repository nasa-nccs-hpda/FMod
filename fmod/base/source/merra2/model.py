import xarray as xa, pandas as pd
import os, math, numpy as np, shutil
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping, Literal
from fmod.pipeline.stats import StatsAccumulator
from fmod.base.util.dates import drepr, date_list
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

def clear_const_file():
	for vres in ["high", "low"]:
		const_filepath = cache_filepath(VarType.Constant,vres=vres)
		remove_filepath(const_filepath)

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

def acess_data_subset( filepath, **kwargs) -> xa.Dataset:
	roi: Optional[ Dict[str,List[float]] ] = cfg().task.get('roi')
	levels: Optional[ List[float] ] = cfg().task.get('levels')
	dataset: xa.Dataset = subset_datavars( xa.open_dataset(filepath, engine='netcdf4', **kwargs) )
	if levels is not None:
		dataset = dataset.sel(z=levels, method="nearest")
	if roi is not None:
		dataset = dataset.sel( x=slice(*roi['x']), y=slice(*roi['y']) )
	return rename_coords(dataset)

@log_timing
def load_dataset(  d: date, vres: str="high" ) -> xa.Dataset:
	filepath =  cache_filepath( VarType.Dynamic, d, vres )
	return acess_data_subset( filepath)

@log_timing
def load_const_dataset( vres: str = "high" ) -> xa.Dataset:
	filepath =  cache_filepath(VarType.Constant, vres=vres )
	return acess_data_subset( filepath)

class FMBatch:

	@log_timing
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

	@log_timing
	def __init__(self, **kwargs):
		self.vres = kwargs.get('vres', "high" )
		self.current_batch: xa.Dataset = None
		self.current_date = None
		self.days_per_batch = cfg().task.batch_ndays
		self.batch_steps: int = self.days_per_batch * get_steps_per_day()
		self.constants: xa.Dataset = load_const_dataset( **kwargs )
		self.norm_data: Dict[str, xa.Dataset] = load_merra2_norm_data()

	def merge_batch(self, slices: List[xa.Dataset]) -> xa.Dataset:
		dynamics: xa.Dataset = xa.concat(slices, dim="time", coords="minimal")
		merged: xa.Dataset =  xa.merge([dynamics, self.constants], compat='override')
		return merged

	def load_batch(self, d: date, **kwargs) -> xa.Dataset:
		dsets = []
		for day in date_list(d, self.days_per_batch):
			dsets.append( load_dataset(day, self.vres) )
		dset = xa.concat(dsets, dim="time", coords="minimal")
		print("Loaded batch  for date {d}:")
		for k, v in dset.data_vars.items():
			print(f" ***>> {k}{v.dims}: {v.shape}")
		return dset # .rename(time="batch")

	def load(self, d: date) -> xa.Dataset:
		if self.current_date != d:
			self.current_batch = self.load_batch(d)
			self.current_date = d

		return self.current_batch












