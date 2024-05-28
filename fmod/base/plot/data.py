from datetime import datetime
from typing import List, Tuple, Dict

import ipywidgets as ipw
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xa
from matplotlib.image import AxesImage

from fmod.base.plot.widgets import StepSlider
from fmod.base.util.config import cfg
from fmod.controller.dual_trainer import TileGrid
from fmod.data.batch import BatchDataset

def start_date( task_config )-> datetime:
	toks = [ int(tok) for tok in task_config.start_date.split("/") ]
	return  datetime( month=toks[0], day=toks[1], year=toks[2] )
def cscale( pvar: xa.DataArray, stretch: float = 2.0 ) -> Tuple[float,float]:
	meanv, stdv, minv = pvar.values.mean(), pvar.values.std(), pvar.values.min()
	vmin = max( minv, meanv - stretch*stdv )
	vmax = meanv + stretch*stdv
	return vmin, vmax

def norm(array):
	array = array - array.mean()
	array = array / array.std()
	return array

class DataPlot(object):
	ptypes = ["input", "target"]

	def __init__(self, input_dataset:  BatchDataset, target_dataset:  BatchDataset, **kwargs):
		self.input_dataset:  BatchDataset = input_dataset
		self.target_dataset:  BatchDataset = target_dataset
		self.ix: int = kwargs.get('ix',0)
		self.iy: int = kwargs.get('iy',0)
		self.channel_index: int = 0
		self.time_index: int = 0
		fsize: float = kwargs.get('fsize', 6.0)
		self.tile_grid: TileGrid = TileGrid()
		self.sample_input: xa.DataArray = input_dataset.get_current_batch_array()
		self.time_coord: List[datetime] = [ pd.Timestamp(d).to_pydatetime() for d in self.sample_input.coords['time'].values]
		self.channel_coord: List[str] = self.sample_input.coords['channel'].values.tolist()
		print( f"sample_input{self.sample_input.dims}{self.sample_input.shape}, channels = {self.channel_coord}")
		self.tslider: StepSlider = StepSlider('Time:', len(self.time_coord))
		self.cslider: StepSlider = StepSlider('Channel:', len(self.channel_coord))
		self.start_time = cfg().task.start_date
		with plt.ioff():
			self.fig, self.axs = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=[fsize*2,fsize], layout="tight")
			self.fig.suptitle(f'Cape Basin: Tile [{self.iy},{self.ix}]', fontsize=14, va="top", y=1.0)
		self.ims: Dict[int,AxesImage] = {}
		self.tslider.set_callback(self.time_update)
		self.cslider.set_callback(self.channel_update)
		self.origin: Dict[str, int] = self.tile_grid.get_tile_origin(self.ix, self.iy)
		self.start_date = start_date( cfg().task )

	@property
	def channel(self) -> str:
		return  self.channel_coord[ self.channel_index ]

	@property
	def datetime(self) -> str:
		return  self.time_coord[ self.time_index ].strftime("%d/%m/%Y:%H")

	def get_dset(self, icol: int ) -> BatchDataset:
		return self.input_dataset if icol == 0 else self.target_dataset

	def generate_plot( self ):
		for icol in [0,1]:
			ax = self.axs[ icol ]
			dset: BatchDataset = self.get_dset(icol)
			batch: xa.DataArray = dset.get_batch_array( self.origin, self.start_date )
			image: xa.DataArray = norm( batch.isel( channel=self.channel_index, time=self.time_index ).squeeze() )
			if icol in self.ims:
				self.ims[icol].set_data(image.values)
			else:
				vrange = cscale(image, 2.0)
				self.ims[icol] = image.plot.imshow(ax=ax, x="x", y="y", cmap='jet', yincrease=True, vmin=vrange[0], vmax=vrange[1])
			ax.set_title(f" {self.ptypes[icol]} {self.channel}[{self.datetime}]", fontsize=10 )
		self.fig.canvas.draw_idle()

	def time_update(self, tindex: int = 0 ):
		self.time_index = tindex
		self.generate_plot()

	def channel_update(self, cindex: int = 0 ):
		self.channel_index = cindex
		self.generate_plot()

	def plot(self):
		self.generate_plot()
		return ipw.VBox([self.fig.canvas, self.tslider, self.cslider])

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

