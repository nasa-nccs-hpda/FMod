from fmod.base.source.merra2.model import load_const_dataset, load_merra2_norm_data, load_dataset
from fmod.base.util.ops import print_norms, vars3d
from xarray.core.resample import DataArrayResample
import xarray as xa, pandas as pd
import numpy as np
from fmod.base.util.config import cfg
from xarray.core.types import T_DataArray
from fmod.base.util.model import dataset_to_stacked
from typing import List, Union, Tuple, Optional, Dict, Type, Any, Sequence, Mapping, Literal, Hashable
import glob, sys, os, time, traceback
from datetime import date
from fmod.base.util.ops import get_levels_config, increasing, replace_nans
from fmod.base.util.logging import lgm, exception_handled, log_timing
from fmod.base.source.merra2.model import merge_batch
from fmod.base.io.loader import ncFormat
from enum import Enum
from xarray.core.types import InterpOptions
from fmod.pipeline.rescale import DataLoader, QType
np.set_printoptions(precision=3, suppress=False, linewidth=150)

def emag( error: xa.DataArray ) -> float:
	ef = error.values.flatten()
	return np.sqrt( (ef*ef).sum() / ef.size )

class Downscaler(object):

	def __init__(self):
		downscale_method: str = cfg().task.downscale_method.split('.')
		self.model = downscale_method[0]
		self.method = downscale_method[1]
		self.cn: Dict[str,str] = dict( x='x', y='y')

	def process( self, variable: xa.DataArray, target: xa.DataArray, qtype: QType=QType.Intensive) -> Dict[str,xa.DataArray]:
		if self.model == "interp":
			result = self._interpolate( variable, target )
		elif self.model == "sfno":
			result = self._sfno( variable, target )
		else:
			raise Exception( f"Unknown downscaling model '{self.model}'")

		if qtype == QType.Extensive:
			nx,ny = target.coords['x'].size, target.coords['y'].size
			result = result/(nx*ny)

		error = result - target

		print( f"Downscaling({self.model}:{self.method}): cumulative error = {emag(error):.3f}")
		return dict( downscale=result, target=target, error=error)

	def _interpolate(self, variable: xa.DataArray, target: xa.DataArray ) -> xa.DataArray:
		interp_method: InterpOptions = InterpOptions(self.method)
		xc, yc = target.coords[self.cn['x']], target.coords[self.cn['y']]
		varray = variable.interp(x=xc, assume_sorted=True, method=interp_method)
		varray =   varray.interp(y=yc, assume_sorted=True, method=interp_method)
		varray.attrs.update(variable.attrs)
		return varray





