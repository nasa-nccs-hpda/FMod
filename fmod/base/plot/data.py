import math, torch, numpy as np
import xarray as xa
from datetime import datetime
from typing  import List, Tuple, Union, Optional, Dict
from fmod.base.util.ops import xaformat_timedeltas, print_data_column
import matplotlib.ticker as ticker
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as ipw
from fmod.base.util.config import cfg
from torch import nn
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from fmod.controller.dual_trainer import TileGrid
from fmod.data.batch import BatchDataset
from fmod.base.plot.widgets import StepSlider
from fmod.base.util.logging import lgm, exception_handled, log_timing

def rms( dvar: xa.DataArray, **kwargs ) -> float:
	varray: np.ndarray = dvar.isel( **kwargs, missing_dims="ignore", drop=True ).values
	return np.sqrt( np.mean( np.square( varray ) ) )

def rmse( diff: xa.DataArray, **kw ) -> xa.DataArray:
	rms_error = np.array( [ rms(diff, time=iT, **kw) for iT in range(diff.shape[0]) ] )
	return xa.DataArray( rms_error, dims=['time'], coords={'time': diff.time} )

def RMSE( diff: xa.DataArray ) -> float:
	d2: xa.DataArray = diff*diff
	return np.sqrt( d2.mean(keep_attrs=True) )
def cscale( pvar: xa.DataArray, stretch: float = 2.0 ) -> Tuple[float,float]:
	meanv, stdv, minv = pvar.values.mean(), pvar.values.std(), pvar.values.min()
	vmin = max( minv, meanv - stretch*stdv )
	vmax = meanv + stretch*stdv
	return vmin, vmax

class DataPlot(object):

	def __init__(self, input_dataset:  BatchDataset, target_dataset:  BatchDataset, **kwargs):
		self.input_dataset:  BatchDataset = input_dataset
		self.target_dataset:  BatchDataset = target_dataset
		self.channel: int = kwargs.get('channel',0)
		fsize: float = kwargs.get('fsize', 8.0)
		self.tile_grid: TileGrid = TileGrid()
		self.sample_input: xa.DataArray = input_dataset.get_current_batch_array().isel(channel=self.channel)
		self.sample_target: xa.DataArray = target_dataset.get_current_batch_array().isel(channel=self.channel)
		self.time: List[datetime] = [ pd.Timestamp(d).to_pydatetime() for d in self.sample_input.coords['time'].values ]
		self.tslider: StepSlider = StepSlider( 'Time:', len(self.time)  )
		with plt.ioff():
			self.fig, self.axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=[fsize*2,fsize], layout="tight")

		print( f" * sample_input{self.sample_input.dims}{self.sample_input.shape}"    )
		print( f" * sample_target{self.sample_target.dims}{self.sample_target.shape}" )

	def get_dset(self, icol: int ) -> BatchDataset:
		return self.input_dataset if icol == 0 else self.target_dataset

	@property
	def ctime(self) -> datetime:
		return self.time[self.tslider.value]

	def generate_plot( self, ix: int=0, iy: int=0 ):
		for icol in [1,2]:
			ax = self.axs[ icol ]
			rmserror = ""
			origin: Dict[str, int] = self.tile_grid.get_tile_origin( ix, iy )
			dset: BatchDataset = self.get_dset(icol)
			image: xa.DataArray = dset.get_batch_array(origin,self.ctime)
			print(f" * PLOT[{icol}]: image{image.dims}{image.shape}")

			# if icol == ncols-1:
			# 	labels[(irow,icol)] = ['targets','predictions'][irow]
			# 	image = images[ labels[(irow,icol)] ]
			# 	if irow == 0: target = image
			# else:
			# 	labels[(irow,icol)] = ['input', 'upsampled'][irow]
			# 	image = images[ labels[(irow,icol)] ]
			# 	image = image.isel( channel=icol )
			# ax.set_aspect(0.5)
			# vrange = cscale( image, 2.0 )
			# tslice: xa.DataArray = image.isel(time=tslider.value).squeeze(drop=True)
			# ims[(irow,icol)] = tslice.plot.imshow( ax=ax, x="x", y="y", cmap='jet', yincrease=True, vmin=vrange[0], vmax=vrange[1]  )
			# if irow == 1: rmserror = f"{RMSE(tslice - target):.3f}"
			# ax.set_title(f" {labels[(irow,icol)]} {rmserror}")



	#
	# def create_plot_data( inputs: np.ndarray, targets: np.ndarray, predictions: np.ndarray, upsampled: np.ndarray, sample_input: xa.DataArray, sample_target: xa.DataArray ) -> Dict[str,xa.DataArray]:
	#
	# 	print(f"sample_input shape = {sample_input.shape}")
	# 	print(f"sample_target shape = {sample_target.shape}")
	# 	print( f"inputs shape = {inputs.shape}")
	# 	print(f"targets shape = {targets.shape}")
	# 	print(f"predictions shape = {predictions.shape}")
	# 	print(f"upsampled shape = {upsampled.shape}")
	# 	tc, ic = sample_target.coords, sample_input.coords
	#
	# 	return dict(    input=       sample_input.copy(  data=inputs.reshape(sample_input.shape) ),
	# 					targets=     sample_target.copy( data=targets.reshape(sample_target.shape) ),
	# 					predictions= sample_target.copy( data=predictions.reshape(sample_target.shape) ),
	# 					upsampled=   xa.DataArray( upsampled, dims=['time','channel','y','x'], coords=dict(time=tc['time'],channel=ic['channel'],y=tc['y'],x=tc['x'])  ) )
	#
	# @exception_handled
	# def mplplot( self, channel: int, **kwargs ):
	# 	ims, labels, rms_errors, target = {}, {}, {}, ""
	#
	#
	# 		for icol in range(ncols):
	# 			ax = self.axs[ irow, icol ]
	# 			rmserror = ""
	# 			if icol == ncols-1:
	# 				labels[(irow,icol)] = ['targets','predictions'][irow]
	# 				image = images[ labels[(irow,icol)] ]
	# 				if irow == 0: target = image
	# 			else:
	# 				labels[(irow,icol)] = ['input', 'upsampled'][irow]
	# 				image = images[ labels[(irow,icol)] ]
	# 				image = image.isel( channel=icol )
	# 			ax.set_aspect(0.5)
	# 			vrange = cscale( image, 2.0 )
	# 			tslice: xa.DataArray = image.isel(time=tslider.value).squeeze(drop=True)
	# 			ims[(irow,icol)] = tslice.plot.imshow( ax=ax, x="x", y="y", cmap='jet', yincrease=True, vmin=vrange[0], vmax=vrange[1]  )
	# 			if irow == 1: rmserror = f"{RMSE(tslice - target):.3f}"
	# 			ax.set_title(f" {labels[(irow,icol)]} {rmserror}")
	#
	# @exception_handled
	# def time_update(self, sindex: int):
	# 	self.fig.suptitle(f'Timestep: {sindex}', fontsize=10, va="top", y=1.0)
	# 	lgm().log( f"time_update: tindex={sindex}")
	# 	target = None
	# 	for icol in [0, 1]:
	# 		ax1 = self.axs[ icol ]
	# 		rmserror = ""
	# 		if icol == ncols - 1:
	# 			labels[(irow, icol)] = ['targets', 'predictions'][irow]
	# 			image = images[labels[(irow, icol)]]
	# 			if irow == 0: target = image
	# 		else:
	# 			labels[(irow, icol)] = ['input', 'upsampled'][irow]
	# 			image = images[labels[(irow, icol)]]
	# 			image = image.isel(channel=icol)
	# 		tslice1: xa.DataArray =  image.isel( time=sindex, drop=True, missing_dims="ignore").fillna( 0.0 )
	# 		ims[(irow,icol)].set_data( tslice1.values.squeeze() )
	# 		if irow == 1: rmserror = f"{RMSE(tslice1 - target):.3f}"
	# 		ax1.set_title(f"{labels[(irow,icol)]} {rmserror}")
	# 	self.fig.canvas.draw_idle()
	#
	#
	# tslider.set_callback( time_update )
	# self.fig.suptitle(f' ** ', fontsize=10, va="top", y=1.0 )
	# return ipw.VBox([self.fig.canvas,tslider])

