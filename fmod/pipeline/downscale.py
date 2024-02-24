import xarray as xa
import numpy as np
from fmod.base.util.config import cfg
from typing import List, Union, Tuple, Optional, Dict, Type, Any, Sequence, Mapping, Literal, Hashable
import glob, sys, os, time, traceback
from fmod.pipeline.rescale import QType
from fmod.models.sfno.downscale import SHTransform
np.set_printoptions(precision=3, suppress=False, linewidth=150)

def emag( error: xa.DataArray ) -> float:
	ef: np.ndarray = error.values.flatten()
	nn = np.count_nonzero(np.isnan(ef))
	N: int =  ef.size - nn
	return np.sqrt( np.nansum(ef*ef) / N )

class Downscaler(object):
	methods = [ "linear", "nearest", "zero", "slinear", "quadratic", "cubic", "polynomial" ]

	def __init__(self, **kwargs ):
		downscale_method: str = cfg().task.downscale_method.split(':')
		self.model = kwargs.get( 'model',  downscale_method[0] )
		self.method= kwargs.get( 'method', downscale_method[1] )
		self.c: Dict[str,str] = cfg().task.coords
		self.kargs = dict( fill_value="extrapolate" )
		if self.method == "polynomial":
			self.kargs['order'] = kwargs.get( 'order', cfg().task.get('poly_order', 5)  )

	def process( self, variable: xa.DataArray, target: xa.DataArray, ) -> Dict[str,xa.DataArray]:    # qtype: QType=QType.Intensive
		t0 = time.time()
		if self.model == "interp":
			result = self._interpolate( variable, target )
		elif self.model == "sfno":
			result = self._sfno( variable, target )
		else:
			raise Exception( f"Unknown downscaling model '{self.model}'")

		# if qtype == QType.Extensive:
		# 	nx,ny = target.coords['x'].size, target.coords['y'].size
		# 	result = result/(nx*ny)

		error = result - target

		print( f"Downscaling({self.model}:{self.method}): cumulative error = {emag(error):.3f}, time = {(time.time()-t0):.3f} sec")
		return dict( downscale=result, target=target, error=error)

	def _interpolate(self, variable: xa.DataArray, target: xa.DataArray ) -> xa.DataArray:
		varray, nn = variable, np.count_nonzero(np.isnan(variable.values))
		coords =  { self.c[cn]: target.coords[ self.c[cn] ] for cn in ['x','y']  }
		varray = varray.interp( coords, self.method, True, self.kargs )
		varray.attrs.update(variable.attrs)
		return varray

	def _sfno(self, source: xa.DataArray, target: xa.DataArray) -> xa.DataArray:
		ds = SHTransform( source, source, method=self.method )
		return ds.process( source )





