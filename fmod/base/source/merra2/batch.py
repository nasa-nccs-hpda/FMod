import xarray as xa, math
from fmod.base.util.config import cfg
from datetime import date
from fmod.base.util.dates import drepr, date_list
from nvidia.dali import fn
from dali.tensors import TensorCPU, TensorListCPU
from enum import Enum
from typing import Any, Mapping, Sequence, Tuple, Union, List, Dict
from fmod.base.util.ops import format_timedeltas, fmbdir
from fmod.base.util.model import dataset_to_stacked
import numpy as np
import pandas as pd

TimedeltaLike = Any  # Something convertible to pd.Timedelta.
TimedeltaStr = str  # A string convertible to pd.Timedelta.

TargetLeadTimes = Union[
	TimedeltaLike,
	Sequence[TimedeltaLike],
	slice  # with TimedeltaLike as its start and stop.
]

_SEC_PER_HOUR = 3600
_HOUR_PER_DAY = 24
SEC_PER_DAY = _SEC_PER_HOUR * _HOUR_PER_DAY
_AVG_DAY_PER_YEAR = 365.24219
AVG_SEC_PER_YEAR = SEC_PER_DAY * _AVG_DAY_PER_YEAR

DAY_PROGRESS = "day_progress"
YEAR_PROGRESS = "year_progress"

predef_norms = [ 'year_progress', 'year_progress_sin', 'year_progress_cos', 'day_progress', 'day_progress_sin', 'day_progress_cos' ]

def get_timedeltas( dset: xa.Dataset ):
	return format_timedeltas( dset.coords["time"] )

def d2xa( dvals: Dict[str,float] ) -> xa.Dataset:
	return xa.Dataset( {vn: xa.DataArray( np.array(dval) ) for vn, dval in dvals.items()} )

def ds2array( dset: xa.Dataset, **kwargs ) -> xa.DataArray:
	merge_dims = kwargs.get( 'merge_dims', ["level", "time"] )
	sizes: Dict[str,int] = {}
	for vn, v in dset.data_vars.items():
		for cn, c in v.coords.items():
			if cn not in (merge_dims + list(sizes.keys())):
				sizes[ cn ] = c.size
	darray: xa.DataArray = dataset_to_stacked( dset, sizes=sizes, preserved_dims=tuple(sizes.keys()) )
	return darray

def array2tensor( darray: xa.DataArray ) -> TensorCPU:
	return TensorCPU(darray.values)

class ncFormat(Enum):
	Standard = 'standard'
	DALI = 'dali'

class BatchType(Enum):
	Training = 'training'
	Forecast = 'forecast'

class VarType(Enum):
	Constant = 'constant'
	Dynamic = 'dynamic'

def suffix( ncformat: ncFormat ) -> str:
	return ".nc" if ncformat == ncformat.Standard else ".dali"

def cache_filepath(vartype: VarType, d: date = None) -> str:
	version = cfg().task.dataset_version
	ncformat: ncFormat = ncFormat( cfg().task.nc_format )
	if vartype == VarType.Dynamic:
		assert d is not None, "cache_filepath: date arg is required for dynamic variables"
		return f"{fmbdir('processed')}/{version}/{drepr(d)}{suffix(ncformat)}"
	else:
		return f"{fmbdir('processed')}/{version}/const{suffix(ncformat)}"

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

def load_dataset( d: date, **kwargs ):
	filepath =  cache_filepath( VarType.Dynamic, d )
	return _open_dataset( filepath, **kwargs)

def load_const_dataset( **kwargs ):
	filepath =  cache_filepath(VarType.Constant)
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

def load_batch( d: date, device: str, **kwargs ):
	filepath = cache_filepath(VarType.Dynamic, d)
	header: xa.Dataset = xa.open_dataset(filepath + "/header.nc", **kwargs)
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

def extract_input_target_times( dataset: xa.Dataset, input_duration: TimedeltaLike, target_lead_times: TargetLeadTimes ) -> Tuple[xa.Dataset, xa.Dataset]:
	"""Extracts inputs and targets for prediction, from a Dataset with a time dim.

	The input period is assumed to be contiguous (specified by a duration), but
	the targets can be a list of arbitrary lead times.

	Examples:

	  # Use 18 hours of data as inputs, and two specific lead times as targets:
	  # 3 days and 5 days after the final input.
	  extract_inputs_targets(
		  dataset,
		  input_duration='18h',
		  target_lead_times=('3d', '5d')
	  )

	  # Use 1 day of data as input, and all lead times between 6 hours and
	  # 24 hours inclusive as targets. Demonstrates a friendlier supported string
	  # syntax.
	  extract_inputs_targets(
		  dataset,
		  input_duration='1 day',
		  target_lead_times=slice('6 hours', '24 hours')
	  )

	  # Just use a single target lead time of 3 days:
	  extract_inputs_targets(
		  dataset,
		  input_duration='24h',
		  target_lead_times='3d'
	  )

	Args:
	  dataset: An xa.Dataset with a 'time' dimension whose coordinates are
		timedeltas. It's assumed that the time coordinates have a fixed offset /
		time resolution, and that the input_duration and target_lead_times are
		multiples of this.
	  input_duration: pandas.Timedelta or something convertible to it (e.g. a
		shorthand string like '6h' or '5d12h').
	  target_lead_times: Either a single lead time, a slice with start and stop
		(inclusive) lead times, or a sequence of lead times. Lead times should be
		Timedeltas (or something convertible to). They are given relative to the
		final input timestep, and should be positive.

	Returns:
	  inputs:
	  targets:
		Two datasets with the same shape as the input dataset except that a
		selection has been made from the time axis, and the origin of the
		time coordinate will be shifted to refer to lead times relative to the
		final input timestep. So for inputs the times will end at lead time 0,
		for targets the time coordinates will refer to the lead times requested.
	"""

	( target_lead_times, target_duration ) = _process_target_lead_times_and_get_duration(target_lead_times)

	# Shift the coordinates for the time axis so that a timedelta of zero
	# corresponds to the forecast reference time. That is, the final timestep
	# that's available as input to the forecast, with all following timesteps
	# forming the target period which needs to be predicted.
	# This means the time coordinates are now forecast lead times.
	time: xa.DataArray = dataset.coords["time"]
	dataset = dataset.assign_coords(time=time + target_duration - time[-1])
	targets: xa.Dataset = dataset.sel({"time": target_lead_times})
	input_duration = pd.Timedelta(input_duration)
	# Both endpoints are inclusive with label-based slicing, so we offset by a
	# small epsilon to make one of the endpoints non-inclusive:
	zero = pd.Timedelta(0)
	epsilon = pd.Timedelta(1, "ns")
	inputs: xa.Dataset = dataset.sel({"time": slice(-input_duration + epsilon, zero)})
	return inputs, targets

def _process_target_lead_times_and_get_duration( target_lead_times: TargetLeadTimes) -> TimedeltaLike:
	"""Returns the minimum duration for the target lead times."""
	# print( f"process_target_lead_times: {target_lead_times}")
	if isinstance(target_lead_times, slice):
		# A slice of lead times. xa already accepts timedelta-like values for
		# the begin/end/step of the slice.
		if target_lead_times.start is None:
			# If the start isn't specified, we assume it starts at the next timestep
			# after lead time 0 (lead time 0 is the final input timestep):
			target_lead_times = slice( pd.Timedelta(1, "ns"), target_lead_times.stop, target_lead_times.step )
		target_duration = pd.Timedelta(target_lead_times.stop)
	else:
		if not isinstance(target_lead_times, (list, tuple, set)):
			# A single lead time, which we wrap as a length-1 array to ensure there
			# still remains a time dimension (here of length 1) for consistency.
			target_lead_times = [target_lead_times]

		# A list of multiple (not necessarily contiguous) lead times:
		target_lead_times = [pd.Timedelta(x) for x in target_lead_times]
		target_lead_times.sort()
		target_duration = target_lead_times[-1]
	return target_lead_times, target_duration


def extract_inputs_targets_forcings( idataset: xa.Dataset, *, input_variables: Tuple[str, ...], target_variables: Tuple[str, ...], forcing_variables: Tuple[str, ...],
	   levels: Tuple[int, ...], input_duration: TimedeltaLike, target_lead_times: TargetLeadTimes, **kwargs ) -> Tuple[TensorCPU, TensorCPU, TensorCPU]:
	idataset = idataset.sel(level=list(levels))
	dvars = {}
	for vname, varray in idataset.data_vars.items():
		missing_batch = ("time" in varray.dims) and ("batch" not in varray.dims)
		dvars[vname] = varray.expand_dims("batch") if missing_batch else varray
	dataset = xa.Dataset( dvars, coords=idataset.coords, attrs=idataset.attrs )
	dataset = dataset.drop_vars("datetime")
	inputs, targets = extract_input_target_times( dataset, input_duration=input_duration, target_lead_times=target_lead_times )
	print( f"\nExtract Inputs & Targets: input times: {get_timedeltas(inputs)}, target times: {get_timedeltas(targets)}")

	if set(forcing_variables) & set(target_variables):
		raise ValueError( f"Forcing variables {forcing_variables} should not overlap with target variables {target_variables}." )

	inputs   = ds2array( inputs[list(input_variables)] )
	print( f" >> inputs{inputs.dims}: {inputs.shape}")
	targets =  ds2array( targets[list(target_variables)] )
	print(f" >> targets{targets.dims}: {targets.shape}")
	forcings = ds2array( targets[list(forcing_variables)] )
	print(f" >> forcings{forcings.dims}: {forcings.shape}")

	return array2tensor(inputs), array2tensor(targets), array2tensor(forcings)
