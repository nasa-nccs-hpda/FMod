import xarray as xa, math, os
from fmod.base.util.config import cfg
from datetime import date
from fmod.base.util.dates import drepr, date_list
from nvidia.dali import fn
from enum import Enum
from fmod.base.io.loader import TSet
from fmod.base.util.config import get_data_indices, get_roi, get_dims
from fmod.controller.stats import StatsAccumulator, StatsEntry
from typing import Any, Mapping, Sequence, Tuple, Union, List, Dict, Literal, Optional
from fmod.base.util.ops import format_timedeltas
from fmod.base.io.loader import data_suffix, path_suffix
from fmod.base.util.logging import lgm, exception_handled, log_timing
import numpy as np
from fmod.base.source.batch import VarType

def d2xa( dvals: Dict[str,float] ) -> xa.Dataset:
	return xa.Dataset( {vn: xa.DataArray( np.array(dval) ) for vn, dval in dvals.items()} )

class Merra2DataLoader(object):

	def __init__(self, vres: str = "high"):
		self.vres = vres

	def cache_filepath(self, vartype: VarType, d: date = None, **kwargs) -> str:
		version = cfg().task.dataset_version
		if vartype == VarType.Dynamic:
			assert d is not None, "cache_filepath: date arg is required for dynamic variables"
			fpath = f"{fmbdir('processed')}/{version}/{drepr(d)}{data_suffix(self.vres)}"
		else:
			fpath = f"{fmbdir('processed')}/{version}/const{data_suffix(self.vres)}"
		os.makedirs(os.path.dirname(fpath), mode=0o777, exist_ok=True)
		return fpath

	def clear_const_file(self):
		for vres in ["high", "low"]:
			const_filepath = self.cache_filepath(VarType.Constant, vres=vres)
			self.remove_filepath(const_filepath)

	@classmethod
	def rename_vars(cls, dataset: xa.Dataset) -> xa.Dataset:
		model_varname_map, model_coord_map = {}, {}
		if 'input_variables' in cfg().task:
			model_varname_map = {v: k for k, v in cfg().task['input_variables'].items() if v in dataset.data_vars}
		if 'coords' in cfg().task:
			model_coord_map = {k: v for k, v in cfg().task['coords'].items() if k in dataset.coords}
		return dataset.rename(**model_varname_map, **model_coord_map)

	def _open_dataset(self, filepath, **kwargs) -> xa.Dataset:
		dataset: xa.Dataset = xa.open_dataset(filepath, **kwargs)
		return self.rename_vars(dataset)

	@classmethod
	def subset_datavars(cls, dataset: xa.Dataset) -> xa.Dataset:
		data_vars = {k: dataset.data_vars[v] for k, v in cfg().task['input_variables'].items() if v in dataset.data_vars}
		return xa.Dataset(data_vars=data_vars, coords=dataset.coords, attrs=dataset.attrs)

	@classmethod
	def rename_coords(cls, dataset: xa.Dataset) -> xa.Dataset:
		model_coord_map = {}, {}
		if 'coords' in cfg().task:
			model_coord_map = {k: v for k, v in cfg().task['coords'].items() if k in dataset.coords}
		return dataset.rename(**model_coord_map)

	@classmethod
	def get_predef_norm_data(cls) -> Dict[str, xa.Dataset]:
		sndef = {sn: sn for sn in StatsAccumulator.statnames}
		snames: Dict[str, str] = cfg().task.get('statnames', sndef)
		dstd = dict(year_progress=0.0247, year_progress_sin=0.003, year_progress_cos=0.003, day_progress=0.433, day_progress_sin=1.0, day_progress_cos=1.0)
		vmean = dict(year_progress=0.5, year_progress_sin=0.0, year_progress_cos=0.0, day_progress=0.5, day_progress_sin=0.0, day_progress_cos=0.0)
		vstd = dict(year_progress=0.29, year_progress_sin=0.707, year_progress_cos=0.707, day_progress=0.29, day_progress_sin=0.707, day_progress_cos=0.707)
		pdstats = dict(std_diff=d2xa(dstd), mean=d2xa(vmean), std=d2xa(vstd))
		return {snames[sname]: pdstats[sname] for sname in sndef.keys()}

	def stats_filepath(self, version: str, statname: str) -> str:
		return f"{fmbdir('processed')}/{version}/stats{path_suffix(self.vres)}/{statname}"

	@log_timing
	def load_stats(self,statname: str, **kwargs) -> xa.Dataset:
		version = cfg().task['dataset_version']
		filepath = self.stats_filepath(version, statname)
		varstats: xa.Dataset = xa.open_dataset(filepath, **kwargs)
		return self.rename_vars(varstats)

	@log_timing
	def load_norm_data(self) -> Dict[str, xa.Dataset]:
		sndef = {sn: sn for sn in StatsAccumulator.statnames}
		snames: Dict[str, str] = cfg().task.get('statnames', sndef)
		return {} if snames is None else {snames[sname]: self.load_stats(sname) for sname in sndef.keys()}

	@log_timing
	def load_merra2_norm_data(self) -> Dict[str, xa.Dataset]:
		predef_norm_data: Dict[str, xa.Dataset] = self.get_predef_norm_data()
		m2_norm_data: Dict[str, xa.Dataset] = self.load_norm_data()
		lgm().log(f"predef_norm_data: {list(predef_norm_data.keys())}")
		lgm().log(f"m2_norm_data: {list(m2_norm_data.keys())}")
		return {nnorm: xa.merge([predef_norm_data[nnorm], m2_norm_data[nnorm]]) for nnorm in m2_norm_data.keys()}

	@classmethod
	def rcoords(cls, dset: xa.Dataset):
		c = dset.coords
		return '[' + ','.join([f"{k}:{c[k].size}" for k in c.keys()]) + ']'

	@classmethod
	def bounds(cls, dset: xa.Dataset):
		c = dset.coords
		dims = get_dims(c)
		return '[' + ','.join([f"{k}[{c[k].size}]:[{c[k][0]:.2f},{c[k][-1]:.2f}:{c[k][1] - c[k][0]:.2f}]" for k in dims]) + ']'

	def access_data_subset(self,filepath, vres: str) -> xa.Dataset:
		levels: Optional[List[float]] = cfg().task.get('levels')
		dataset: xa.Dataset = self.subset_datavars(xa.open_dataset(filepath, engine='netcdf4'))
		if (levels is not None) and ('z' in dataset.coords):
			dataset = dataset.sel(z=levels, method="nearest")
		lgm().log(f"LOAD[{vres}]-> dims: {self.rcoords(dataset)}")
		iorigin: Dict[str, int] = get_data_indices(dataset, cfg().task.origin[ TSet.Train.value ] )
		tile_size: Dict[str, int] = cfg().task.tile_size

		if vres == "high":
			iextent: Dict[str, int] = get_data_indices(dataset, cfg().task.extent)
			iroi = {dim: slice(oidx, iextent[dim]) for dim, oidx in iorigin.items()}
		elif vres == "low":
			iroi = {dim: slice(oidx, oidx + tile_size[dim]) for dim, oidx in iorigin.items()}
		else:
			raise Exception(f"Unrecognized vres: {vres}")

		dataset = dataset.isel(**iroi)
		lgm().log(f" %% data_subset[{vres}]-> iroi: {iroi}, dataset roi: {get_roi(dataset.coords)}")
		return self.rename_coords(dataset)

	def load_dataset(self, d: date) -> xa.Dataset:
		filepath = self.cache_filepath(VarType.Dynamic, d)
		result: xa.Dataset = self.access_data_subset(filepath, self.vres)
		lgm().log(f" * load_dataset[{self.vres}]({d}) {self.bounds(result)} nts={result.coords['time'].size} {filepath}")
		return result

	def load_const_dataset(self) -> xa.Dataset:
		filepath = self.cache_filepath(VarType.Constant)
		return self.access_data_subset(filepath, self.vres)

	# def load_batch(self, d: date, **kwargs):
	# 	filepath = self.cache_filepath(VarType.Dynamic, d)
	# 	device = cfg().task.device.lower()
	# 	header: xa.Dataset = xa.open_dataset(filepath + "/header.nc", engine="netcdf4", **kwargs)
	# 	files: List[str] = [f"{vn}.npy" for vn in header.attrs['data_vars']]
	# 	coords: Mapping[str, xa.DataArray] = header.data_vars
	# 	data = fn.readers.numpy(device=device, file_root=filepath, files=files)
	# 	print(f"loaded {len(files)}, result = {type(data)}")
	# 	return data