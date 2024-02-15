import xarray as xa, math, os
from fmod.base.util.config import cfg
from datetime import date
from fmod.base.util.dates import drepr, date_list
from nvidia.dali import fn
from enum import Enum
from typing import Any, Mapping, Sequence, Tuple, Union, List, Dict
from fmod.base.util.ops import format_timedeltas, fmbdir
import numpy as np

predef_norms = [ 'year_progress', 'year_progress_sin', 'year_progress_cos', 'day_progress', 'day_progress_sin', 'day_progress_cos' ]
_SEC_PER_HOUR = 3600
_HOUR_PER_DAY = 24
SEC_PER_DAY = _SEC_PER_HOUR * _HOUR_PER_DAY
_AVG_DAY_PER_YEAR = 365.24219
AVG_SEC_PER_YEAR = SEC_PER_DAY * _AVG_DAY_PER_YEAR

DAY_PROGRESS = "day_progress"
YEAR_PROGRESS = "year_progress"

class ncFormat(Enum):
	Standard = 'standard'
	DALI = 'dali'
	SRES = "sres"

class BatchType(Enum):
	Training = 'training'
	Forecast = 'forecast'

class VarType(Enum):
	Constant = 'constant'
	Dynamic = 'dynamic'

def suffix( vres: str, ncformat: ncFormat ) -> str:
	res_suffix = "" if vres == "high" else "." + vres
	format_suffix = ".nc" if ncformat == ncformat.Standard else ".dali"
	return res_suffix + format_suffix

def cache_filepath(vartype: VarType, vres: str, d: date = None) -> str:
	version = cfg().task.dataset_version
	ncformat: ncFormat = ncFormat( cfg().task.nc_format )
	if vartype == VarType.Dynamic:
		assert d is not None, "cache_filepath: date arg is required for dynamic variables"
		fpath = f"{fmbdir('processed')}/{version}/{drepr(d)}{suffix(vres,ncformat)}"
	else:
		fpath = f"{fmbdir('processed')}/{version}/const{suffix(vres,ncformat)}"
	os.makedirs(os.path.dirname(fpath), mode=0o777, exist_ok=True)
	return fpath

def stats_filepath(version: str, statname: str) -> str:
	return f"{fmbdir('processed')}/{version}/stats/{statname}"
def get_target_steps(btype: BatchType):
	if btype == BatchType.Training: return cfg().task['train_steps']
	elif btype == BatchType.Forecast: return cfg().task['eval_steps']

def get_days_per_batch(btype: BatchType):
	steps_per_day: float = 24 / cfg().task['data_timestep']
	assert steps_per_day.is_integer(), "steps_per_day (24/data_timestep) must be an integer"
	target_steps = get_target_steps( btype )
	batch_steps: int = cfg().task['input_steps'] + target_steps
	if btype == BatchType.Training: return 1 + math.ceil((batch_steps - 1) / steps_per_day)
	elif btype == BatchType.Forecast: return math.ceil(batch_steps / steps_per_day)
def rename_vars( dataset: xa.Dataset ) -> xa.Dataset:
	model_varname_map, model_coord_map = {}, {}
	if 'input_variables' in cfg().task:
		model_varname_map = {v: k for k, v in cfg().task['input_variables'].items() if v in dataset.data_vars}
	if 'coords' in cfg().task:
		model_coord_map = {k: v for k, v in cfg().task['coords'].items() if k in dataset.coords}
	return dataset.rename(**model_varname_map, **model_coord_map)

def _open_dataset( filepath, **kwargs) -> xa.Dataset:
	dataset: xa.Dataset = xa.open_dataset(filepath, **kwargs)
	return rename_vars(dataset)

def load_dataset( d: date, vres: str, **kwargs ):
	filepath =  cache_filepath( VarType.Dynamic, vres, d )
	return _open_dataset( filepath, **kwargs)

def load_const_dataset(  vres: str, **kwargs ):
	filepath =  cache_filepath(VarType.Constant, vres)
	return _open_dataset( filepath, **kwargs )

def merge_batch( self, slices: List[xa.Dataset], constants: xa.Dataset ) -> xa.Dataset:
	constant_vars: List[str] = self.task_config.get('constants',[])
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

def load_batch( d: date, **kwargs ):
	filepath = cache_filepath(VarType.Dynamic, d)
	device = cfg().task.device.lower()
	header: xa.Dataset = xa.open_dataset(filepath + "/header.nc", engine="netcdf4", **kwargs)
	files: List[str] = [ f"{vn}.npy" for vn in header.attrs['data_vars'] ]
	coords: Mapping[str, xa.DataArray] = header.data_vars
	data = fn.readers.numpy(device=device, file_root=filepath, files=files)
	print( f"loaded {len(files)}, result = {type(data)}")
	return data

def load_predef_norm_data() -> Dict[str,xa.Dataset]:
	root, norms, drop_vars = fmbdir('model'), {}, None
	with open(f"{root}/stats/diffs_stddev_by_level.nc", "rb") as f:
		dset: xa.Dataset = xa.load_dataset(f)
		drop_vars = [ vname for vname in dset.data_vars.keys() if vname not in predef_norms ]
		norms['diffs_stddev_by_level']: xa.Dataset = dset.drop_vars( drop_vars ).compute()
	with open(f"{root}/stats/mean_by_level.nc", "rb") as f:
		dset: xa.Dataset = xa.load_dataset(f)
		drop_vars = [ vname for vname in dset.data_vars.keys() if vname not in predef_norms ]
		norms['mean_by_level']: xa.Dataset = dset.drop_vars( drop_vars ).compute()
	with open(f"{root}/stats/stddev_by_level.nc", "rb") as f:
		dset: xa.Dataset = xa.load_dataset(f)
		drop_vars = [ vname for vname in dset.data_vars.keys() if vname not in predef_norms ]
		norms['stddev_by_level']: xa.Dataset = dset.drop_vars( drop_vars ).compute()
	for nname, norm in norms.items():
		print( f" __________________ {nname} __________________ ")
		for (vname,darray) in norm.data_vars.items():
			print( f"   > {vname}: dims={darray.dims}, shape={darray.shape}, coords={list(darray.coords.keys())}  ")
	return norms

def get_year_progress(seconds_since_epoch: np.ndarray) -> np.ndarray:
	"""Computes year progress for times in seconds.

	Args:
	  seconds_since_epoch: Times in seconds since the "epoch" (the point at which
		UNIX time starts).

	Returns:
	  Year progress normalized to be in the [0, 1) interval for each time point.
	"""

	# Start with the pure integer division, and then float at the very end.
	# We will try to keep as much precision as possible.
	years_since_epoch = ( seconds_since_epoch / SEC_PER_DAY / np.float64(_AVG_DAY_PER_YEAR) )
	# Note depending on how these ops are down, we may end up with a "weak_type"
	# which can cause issues in subtle ways, and hard to track here.
	# In any case, casting to float32 should get rid of the weak type.
	# [0, 1.) Interval.
	yp = np.mod(years_since_epoch, 1.0).astype(np.float32)
	return yp


def get_day_progress( seconds_since_epoch: np.ndarray, longitude: np.ndarray ) -> np.ndarray:
	"""Computes day progress for times in seconds at each longitude.

	Args:
	  seconds_since_epoch: 1D array of times in seconds since the 'epoch' (the
		point at which UNIX time starts).
	  longitude: 1D array of longitudes at which day progress is computed.

	Returns:
	  2D array of day progress values normalized to be in the [0, 1) inverval
		for each time point at each longitude.
	"""

	# [0.0, 1.0) Interval.
	day_progress_greenwich = ( np.mod(seconds_since_epoch, SEC_PER_DAY) / SEC_PER_DAY )

	# Offset the day progress to the longitude of each point on Earth.
	longitude_offsets = np.deg2rad(longitude) / (2 * np.pi)
	day_progress = np.mod( day_progress_greenwich[..., np.newaxis] + longitude_offsets, 1.0 )
	return day_progress.astype(np.float32)


def featurize_progress( name: str, dims: Sequence[str], progress: np.ndarray ) -> Mapping[str, xa.Variable]:
	"""Derives features used by ML models from the `progress` variable.

	Args:
	  name: Base variable name from which features are derived.
	  fdims: List of the output feature dimensions, e.g. ("day", "lon").
	  progress: Progress variable values.

	Returns:
	  Dictionary of xa variables derived from the `progress` values. It
	  includes the original `progress` variable along with its sin and cos
	  transformations.

	Raises:
	  ValueError if the number of feature dimensions is not equal to the number
		of data dimensions.
	"""
	if len(dims) != progress.ndim:
		raise ValueError( f"Number of dimensions in feature {name}{dims} must be equal to the number of dimensions in progress{progress.shape}." )
	else: print( f"featurize_progress: {name}{dims} --> progress{progress.shape} ")

	progress_phase = progress * (2 * np.pi)
	return {
		name: xa.Variable(dims, progress),
		name + "_sin": xa.Variable(dims, np.sin(progress_phase)),
		name + "_cos": xa.Variable(dims, np.cos(progress_phase)),
	}


