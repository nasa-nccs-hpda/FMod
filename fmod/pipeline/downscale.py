import xarray as xa
import numpy as np
from fmod.base.util.config import cfg
from typing import List, Union, Tuple, Optional, Dict, Type, Any, Sequence, Mapping, Literal, Hashable
import glob, sys, os, time, traceback
from fmod.pipeline.rescale import QType
np.set_printoptions(precision=3, suppress=False, linewidth=150)

def emag( error: xa.DataArray ) -> float:
	ef: np.ndarray = error.values.flatten()
	nn = np.count_nonzero(np.isnan(ef))
	N: int =  ef.size - nn
	print( f" ------------------------ emag: N={N}, nn={nn}, efmax={np.nanmax(ef)}, efmin={np.nanmin(ef)}")
	return np.sqrt( np.nansum(ef*ef) / N )

class Downscaler(object):

	def __init__(self, **kwargs ):
		downscale_method: str = cfg().task.downscale_method.split(':')
		self.model = kwargs.get( 'model',  downscale_method[0] )
		self.method= kwargs.get( 'method', downscale_method[1] )
		self.c: Dict[str,str] = cfg().task.coords
		self.order = kwargs.get( 'order', cfg().task.get('poly_order', 2)  )
		self.kargs = dict(assume_sorted=True)
		if self.method == "polynomial":
			self.kargs['kwargs'] =  dict( order=self.order )

	def process( self, variable: xa.DataArray, target: xa.DataArray, qtype: QType=QType.Intensive) -> Dict[str,xa.DataArray]:
		t0 = time.time()
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

		print( f"Downscaling({self.model}:{self.method}): cumulative error = {emag(error):.2f}, time = {(time.time()-t0):.2f} sec")
		return dict( downscale=result, target=target, error=error)

	def _interpolate(self, variable: xa.DataArray, target: xa.DataArray ) -> xa.DataArray:
		varray, nn = variable, np.count_nonzero(np.isnan(variable.values))
		print( f" ****************** Interpolating({self.method}) with nn={nn}")
		coords =  { self.c[cn]: target.coords[ self.c[cn] ] for cn in ['x','y']  }
		varray = varray.interp( coords, self.method, **self.kargs )
		varray.attrs.update(variable.attrs)
		return varray

	def _sfno(self, variable: xa.DataArray, target: xa.DataArray) -> xa.DataArray:
		return target





