import xarray as xa, math
from fmod.base.util.config import cfg
from datetime import date
from typing  import List, Tuple, Union, Optional, Mapping
from fmod.base.util.dates import drepr, date_list
from nvidia.dali import fn
from enum import Enum
from fmod.base.util.ops import fmbdir
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

def load_batch( d: date, **kwargs ):
	filepath = cache_filepath(VarType.Dynamic, d)
	header: xa.Dataset = xa.open_dataset(filepath + "/header.nc", **kwargs)
	files: List[str] = [ f"{vn}.npy" for vn in header.attrs['data_vars'] ]
	coords: Mapping[str, xa.DataArray] = header.data_vars
	data = fn.readers.numpy(device='gpu', file_root=filepath, files=files)
	print( f"loaded {len(files)}, result = {type(data)}")