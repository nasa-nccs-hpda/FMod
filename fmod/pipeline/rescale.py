from fmod.base.source.merra2.model import load_const_dataset, load_merra2_norm_data, load_dataset
import xarray as xa, pandas as pd
import os, math, numpy as np, shutil
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping
from fmod.pipeline.stats import StatsAccumulator
from fmod.base.util.dates import drepr, date_list
from datetime import date
from fmod.base.util.logging import lgm, exception_handled, log_timing
from fmod.base.util.config import cfg
from fmod.base.source.merra2.batch import rename_vars, get_days_per_batch, get_target_steps, VarType, BatchType, cache_filepath, stats_filepath
from fmod.base.io.loader import ncFormat
from fmod.base.util.ops import print_norms, vars3d
from xarray.core.resample import DataArrayResample
import xarray as xa, pandas as pd
import numpy as np
from fmod.base.util.config import cfg
from typing import List, Union, Tuple, Optional, Dict, Type, Any, Sequence, Mapping
import glob, sys, os, time, traceback
from fmod.base.util.ops import fmbdir
from fmod.base.util.dates import skw, dstr
from datetime import date
from fmod.base.util.ops import get_levels_config, increasing, replace_nans
np.set_printoptions(precision=3, suppress=False, linewidth=150)
from numpy.lib.format import write_array
from fmod.base.util.logging import lgm, exception_handled, log_timing
from fmod.pipeline.stats import StatsAccumulator, StatsEntry
from fmod.base.source.merra2.model import merge_batch
from fmod.base.io.loader import ncFormat
from enum import Enum
from xarray.core.types import InterpOptions

class QType(Enum):
	Intensive = 'intensive'
	Extensive = 'extensive'

class DataLoader(object):

	def __init__(self,  **kwargs):
		self.format = ncFormat(cfg().task.get('nc_format', 'standard'))
		self.interp_method =  cfg().task.get('interp_method','linear')
		self.corder = ['time', 'z', 'y', 'x']
		self.xext, self.yext = cfg().preprocess.get('xext'), cfg().preprocess.get('yext')
		self.xres, self.yres = cfg().preprocess.get('xres'), cfg().preprocess.get('yres')
		self.levels: Optional[np.ndarray] = get_levels_config(cfg().preprocess)
		self.tstep = str(cfg().preprocess.data_timestep) + "h"
		self.dmap: Dict = cfg().preprocess.dims
		self.upscale_factor: int = cfg().task.get('upscale_factor')
		self._constant_data = {}
		self.norm_data: Dict[str, xa.Dataset] = load_merra2_norm_data()

	def constant_data(self, vres: str, **kwargs ):
		return self._constant_data.setdefault( vres,  load_const_dataset( vres, **kwargs ) )

	def get_dataset(self, vres: str,  d: date, **kwargs ) -> xa.Dataset:
		time_index = kwargs.pop('time_index',-1)
		dset: xa.Dataset = load_dataset( vres, d ).squeeze( drop=True )
		if time_index >= 0: dset=dset.isel( time=time_index, drop=True )
		merged: xa.Dataset = merge_batch( [ dset ], self.constant_data(vres,**kwargs) )
		return merged

	def get_channel_array(self, vres: str, d: date, **kwargs) -> xa.DataArray:
		dset: xa.Dataset = self.get_dataset( vres, d, **kwargs )
		cvars = kwargs.pop('vars', vars3d(dset))
		dset = dset.drop_vars( set(dset.data_vars.keys()) - set(cvars) )
		return self.to_feature_array( dset )

	@classmethod
	def to_feature_array( cls, data_batch: xa.Dataset) -> xa.DataArray:
		features = xa.Variable('features', list(data_batch.data_vars.keys()) )
	#	feature_index = xa.Index.from_variables( dict(features=features) )
		result: xa.DataArray = xa.concat( list(data_batch.data_vars.values()), dim=features )
		print(f"result shape: {result.shape}, dims: {result.dims}, coord({result.dims[0]}): {result.coords[result.dims[0]].values.tolist()}")
		result = result.transpose(..., "features" )
		return result

	def interp_axis(self, dvar: xa.DataArray, coords: Dict[str, Any], axis: str):
		assert axis in ['x', 'y'], f"Invalid axis: {axis}"
		res, ext = (self.xres, self.xext) if (axis == 'x') else (self.yres, self.yext)
		if res is not None:
			if ext is None:
				c0 = dvar.coords[axis].values
				if axis == 'x':   self.xext = [c0[0], c0[-1]]
				else:           self.yext = [c0[0], c0[-1]]
			ext1 = ext[1] if axis == 'x' else ext[1] + res / 2
			coords[axis] = np.arange(ext[0], ext1, res)
		elif ext is not None:
			coords[axis] = slice(ext[0], ext[1])

	def interp_axes(self, dvar: xa.DataArray, subsample_coords: Dict[str, Dict[str, np.ndarray]], vres: str):
		coords: Dict[str, Any] = subsample_coords.setdefault(vres, {})
		if (self.levels is not None) and ('z' in dvar.dims):
			coords['z'] = self.levels
		for axis in ['x', 'y']:
			if vres == "high":
				self.interp_axis(dvar, coords, axis)
			else:
				hres_coords: Dict[str, np.ndarray] = subsample_coords['high']
				hres_axis = hres_coords[axis] if axis in hres_coords else dvar.coords[axis]
				coords[axis] = hres_axis[0::self.upscale_factor]

	def subsample_coords(self, dvar: xa.DataArray) -> Dict[str, Dict[str, np.ndarray]]:
		sscoords: Dict[str, Dict[str, np.ndarray]] = {}
		for vres in ["high", "low"]:
			if vres == "high" or self.format == ncFormat.SRES:
				self.interp_axes(dvar, sscoords, vres)
		return sscoords

	def subsample(self, variable: xa.DataArray, global_attrs: Dict, qtype: QType, isconst: bool) -> Dict[str, List[xa.DataArray]]:
		ssvars: Dict[str, List] = {}
		cmap: Dict[str, str] = {cn0: cn1 for (cn0, cn1) in self.dmap.items() if cn0 in list(variable.coords.keys())}
		variable: xa.DataArray = variable.rename(**cmap)
		if isconst and ("time" in variable.dims):
			variable = variable.isel(time=0, drop=True)
		sscoords: Dict[str, Dict[str, np.ndarray]] = self.subsample_coords(variable)
		for vres, vcoord in sscoords.items():
			svars = ssvars.setdefault(vres, [])
			lgm().log(f" **** subsample {variable.name}:{vres}, vc={list(vcoord.keys())}, dims={variable.dims}, shape={variable.shape}, new sizes: { {cn: cv.size for cn, cv in vcoord.items()} }")
			varray: xa.DataArray = self._interp(variable, vcoord, global_attrs, qtype)
			svars.append(varray)
		return ssvars

	def _interp(self, variable: xa.DataArray, vcoord: Dict[str, np.ndarray], global_attrs: Dict, qtype: QType) -> xa.DataArray:
		varray = variable.interp( x=vcoord['x'], assume_sorted=True,  method=self.interp_method ) if 'x' in vcoord else variable
		varray =   varray.interp( y=vcoord['y'], assume_sorted=True,  method=self.interp_method ) if 'y' in vcoord else varray
		varray =   varray.interp( z=vcoord['z'], assume_sorted=False, method=self.interp_method ) if 'z' in vcoord else varray
		if 'time' in varray.dims:
			resampled: DataArrayResample = varray.resample(time=self.tstep)
			varray: xa.DataArray = resampled.mean() if qtype == QType.Intensive else resampled.sum()
		varray.attrs.update(global_attrs)
		varray.attrs.update(varray.attrs)
		for missing in ['fmissing_value', 'missing_value', 'fill_value']:
			if missing in varray.attrs:
				missing_value = varray.attrs.pop('fmissing_value')
				varray = varray.where(varray != missing_value, np.nan)
		return replace_nans(varray).transpose(*self.corder, missing_dims="ignore")