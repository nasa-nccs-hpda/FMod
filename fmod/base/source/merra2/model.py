import xarray as xa, pandas as pd
import os, math, numpy as np, shutil
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping
from fmod.base.source.merra2.preprocess import StatsAccumulator
from fmod.base.util.dates import drepr, date_list
from datetime import date
from fmod.base.util.config import cfg
from .batch import rename_vars, get_days_per_batch, get_target_steps, VarType, BatchType, cache_filepath, stats_filepath, ncFormat

_SEC_PER_HOUR = 3600
_HOUR_PER_DAY = 24
SEC_PER_DAY = _SEC_PER_HOUR * _HOUR_PER_DAY
_AVG_DAY_PER_YEAR = 365.24219
AVG_SEC_PER_YEAR = SEC_PER_DAY * _AVG_DAY_PER_YEAR

def nnan(varray: xa.DataArray) -> int: return np.count_nonzero(np.isnan(varray.values))
def pctnan(varray: xa.DataArray) -> str: return f"{nnan(varray)*100.0/varray.size:.2f}%"

def d2xa( dvals: Dict[str,float] ) -> xa.Dataset:
    return xa.Dataset( {vn: xa.DataArray( np.array(dval) ) for vn, dval in dvals.items()} )

def clear_const_file():
	const_filepath = cache_filepath(VarType.Constant)
	if os.path.exists(const_filepath):
		try: os.remove(const_filepath)
		except IsADirectoryError:
			shutil.rmtree(const_filepath)

def merge_batch( slices: List[xa.Dataset], constants: xa.Dataset ) -> xa.Dataset:
	constant_vars: List[str] = cfg().task.get('constants',[])
	cvars = [vname for vname, vdata in slices[0].data_vars.items() if "time" not in vdata.dims]
	dynamics: xa.Dataset = xa.concat( slices, dim="time", coords = "minimal" )
	dynamics = dynamics.drop_vars(cvars)
	sample: xa.Dataset = slices[0].drop_dims( 'time', errors='ignore' )
	for vname, dvar in sample.data_vars.items():
		if vname not in dynamics.data_vars.keys():
			constants[vname] = dvar
		elif (vname in constant_vars) and ("time" in dvar.dims):
			dvar = dvar.mean(dim="time", skipna=True, keep_attrs=True)
			constants[vname] = dvar
	dynamics = dynamics.drop_vars(constant_vars, errors='ignore')
	return xa.merge( [dynamics, constants], compat='override' )

def get_predef_norm_data() -> Dict[str, xa.Dataset]:
	sndef = { sn:sn for sn in StatsAccumulator.statnames }
	snames: Dict[str, str] = cfg().task.get('statnames',sndef)
	dstd = dict(year_progress=0.0247, year_progress_sin=0.003, year_progress_cos=0.003, day_progress=0.433, day_progress_sin=1.0, day_progress_cos=1.0)
	vmean = dict(year_progress=0.5, year_progress_sin=0.0, year_progress_cos=0.0, day_progress=0.5, day_progress_sin=0.0, day_progress_cos=0.0)
	vstd = dict(year_progress=0.29, year_progress_sin=0.707, year_progress_cos=0.707, day_progress=0.29, day_progress_sin=0.707, day_progress_cos=0.707)
	pdstats = dict(std_diff=d2xa(dstd), mean=d2xa(vmean), std=d2xa(vstd))
	return {snames[sname]: pdstats[sname] for sname in sndef.keys()}

def load_stats(statname: str, **kwargs) -> xa.Dataset:
	version = cfg().task['dataset_version']
	filepath = stats_filepath(version, statname)
	varstats: xa.Dataset = xa.open_dataset(filepath, **kwargs)
	return rename_vars( varstats )

def load_norm_data() -> Dict[str, xa.Dataset]:
	sndef = { sn:sn for sn in StatsAccumulator.statnames }
	snames: Dict[str, str] = cfg().task.get('statnames',sndef)
	return {} if snames is None else {snames[sname]: load_stats(sname) for sname in sndef.keys()}

def load_merra2_norm_data() -> Dict[str, xa.Dataset]:
	predef_norm_data: Dict[str, xa.Dataset] = get_predef_norm_data()
	m2_norm_data: Dict[str, xa.Dataset] = load_norm_data()
	print( f"predef_norm_data: {list(predef_norm_data.keys())}")
	print( f"m2_norm_data: {list(m2_norm_data.keys())}")
	return {nnorm: xa.merge([predef_norm_data[nnorm], m2_norm_data[nnorm]]) for nnorm in m2_norm_data.keys()}

def open_dataset( filepath, **kwargs) -> xa.Dataset:
	dataset: xa.Dataset = xa.open_dataset(filepath, **kwargs)
	return rename_vars(dataset)

def load_dataset(  d: date, **kwargs ):
	filepath =  cache_filepath( VarType.Dynamic, d )
	return open_dataset( filepath, **kwargs)

def load_const_dataset( **kwargs ):
	filepath =  cache_filepath(VarType.Constant)
	return open_dataset( filepath, **kwargs )

class FMBatch:

	def __init__(self, btype: BatchType, **kwargs):
		self.format = ncFormat( cfg().task.get('nc_format', 'standard') )
		self.type: BatchType = btype
		self.days_per_batch = get_days_per_batch(btype)
		self.target_steps = get_target_steps(btype)
		self.batch_steps: int = cfg().task['input_steps'] + self.target_steps
		self.constants: xa.Dataset = load_const_dataset( **kwargs )
		self.norm_data: Dict[str, xa.Dataset] = load_merra2_norm_data()
		self.current_batch: xa.Dataset = None

	def load(self, d: date, **kwargs):
		bdays = date_list(d, self.days_per_batch)
		time_slices: List[xa.Dataset] = [ load_dataset(d, **kwargs) for d in bdays ]
		self.current_batch: xa.Dataset = merge_batch(time_slices, self.constants)

	#	print( f"\n *********** Loaded batch, days_per_batch={self.days_per_batch}, batch_steps={self.batch_steps}, ndays={len(bdays)} *********** " )
	#	print(f" >> times= {[str(Timestamp(t).date()) for t in self.current_batch.coords['time'].values.tolist()]} ")
	#	for vn, dv in self.current_batch.data_vars.items():
	#		print(f" >> {vn}{dv.dims}: {dv.shape}")

	def get_train_data(self,  day_offset: int ) -> xa.Dataset:
		return self.current_batch.isel( time=slice(day_offset, day_offset+self.batch_steps) )

	@classmethod
	def to_feature_array( cls, data_batch: xa.Dataset) -> xa.DataArray:
		features = xa.DataArray(data=list(data_batch.data_vars.keys()), name="features")
		result = xa.concat( list(data_batch.data_vars.values()), dim=features )
		result = result.transpose(..., "features")
		return result






	# def load_timestep( date: date, task: Dict, **kwargs ) -> xa.Dataset:
	# 	vnames = kwargs.pop('vars',None)
	# 	vlist: Dict[str, str] = task['input_variables']
	# 	constants: List[str] = task['constants']
	# 	levels: Optional[np.ndarray] = get_levels_config(task)
	# 	version = task['dataset_version']
	# 	cmap = task['coords']
	# 	zc, yc, corder = cmap['z'], cmap['y'], [ cmap[cn] for cn in ['t','z','y','x'] ]
	# 	tsdata = {}
	# 	filepath = cache_var_filepath(version, date)
	# #	if not os.path.exists( filepath ):
	# 	dataset: xa.Dataset = xa.open_dataset(filepath, **kwargs)
	# 	print(f"  load_timestep({date}), constants={constants}, kwargs={kwargs} ")
	# 	for vname,dsname in vlist.items():
	# 		if (vnames is None) or (vname in vnames):
	# 			varray: xa.DataArray = dataset.data_vars[vname]
	# 			if (vname in constants) and ("time" in varray.dims):
	# 				varray = varray.mean( dim="time", skipna=True, keep_attrs=True )
	# 			varray.attrs['dset_name'] = dsname
	# 			print( f" >> Load_var({dsname}): name={vname}, shape={varray.shape}, dims={varray.dims}, zc={zc}, mean={varray.values.mean()}, nnan={nnan(varray)} ({pctnan(varray)})")
	# 			tsdata[vname] = varray
	# 	return xa.Dataset( tsdata )






