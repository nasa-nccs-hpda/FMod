import math, torch, numpy as np
import xarray as xa
from typing  import List, Tuple, Union, Optional, Dict
from fmod.base.util.ops import xaformat_timedeltas, print_data_column
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import ipywidgets as ipw
from fmod.controller.stats import l2loss
from fmod.base.util.config import cfg
from torch import nn
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from xarray.core.coordinates import DataArrayCoordinates, DatasetCoordinates
from fmod.controller.dual_trainer import ModelTrainer
from fmod.controller.dual_trainer import LearningContext
from fmod.view.tile_selection_grid import TileSelectionGrid
from fmod.base.plot.widgets import StepSlider
from fmod.base.util.logging import lgm, exception_handled, log_timing

colors = ["red", "blue", "green", "cyan", "magenta", "yellow", "grey", "brown", "pink", "purple", "orange", "black"]

def flex(weight: int) -> ipw.Layout:
	return ipw.Layout(flex=f'1 {weight} auto', width='auto')
def rms( dvar: xa.DataArray, **kwargs ) -> float:
	varray: np.ndarray = dvar.isel( **kwargs, missing_dims="ignore", drop=True ).values
	return np.sqrt( np.mean( np.square( varray ) ) )

def tensor( dvar: xa.DataArray ) -> torch.Tensor:
	return torch.from_numpy( dvar.values.squeeze() )

def rmse( diff: xa.DataArray, **kw ) -> xa.DataArray:
	rms_error = np.array( [ rms(diff, time=iT, **kw) for iT in range(diff.shape[0]) ] )
	return xa.DataArray( rms_error, dims=['time'], coords={'time': diff.time} )

def cscale( pvar: xa.DataArray, stretch: float = 2.0 ) -> Tuple[float,float]:
	meanv, stdv, minv = pvar.values.mean(), pvar.values.std(), pvar.values.min()
	vmin = max( minv, meanv - stretch*stdv )
	vmax = meanv + stretch*stdv
	return vmin, vmax

def normalize( target: xa.Dataset, vname: str, **kwargs ) -> xa.DataArray:
	statnames: Dict[str,str] = kwargs.get('statnames', dict(mean='mean', std='std'))
	norms: Dict[str,xa.Dataset] = kwargs.pop( 'norms', {} )
	fvar: xa.DataArray = target.data_vars[vname]
	if 'batch' in fvar.dims:  fvar = fvar.squeeze(dim="batch", drop=True)
	if len(norms) == 0: return fvar
	stats: Dict[str,xa.DataArray] = { stat: statdata.data_vars[vname] for stat,statdata in norms.items()}
	return (fvar-stats[ statnames['mean'] ]) / stats[ statnames['std'] ]

def to_xa( template: xa.DataArray, data: np.ndarray ) -> xa.DataArray:
	return template.copy(data=data.reshape(template.shape))

def onclick(event):
	lgm().log('Mouse click: button=%d, dbl=%d, x=%d, y=%d, xdata=%f, ydata=%f' % (event.button, event.dblclick, event.x, event.y, event.xdata, event.ydata), display=True)

class SRPlot(object):
	def __init__(self, trainer: ModelTrainer, context: LearningContext, **kwargs):
		self.trainer: ModelTrainer = trainer
		self.context: LearningContext = context
		self.channel = kwargs.get('channel', 0)
		self.splabels = [['input', self.upscale_plot_label], ['target', self.result_plot_label]]
		self.sample_input: xa.DataArray  = trainer.get_sample_input()
		self.sample_target: xa.DataArray = trainer.get_sample_target()
		self.input: xa.DataArray = to_xa( self.sample_input, trainer.get_ml_input(context) )
		self.target: xa.DataArray = to_xa( self.sample_target, trainer.get_ml_target(context) )
		self.prediction: xa.DataArray = to_xa( self.sample_target, trainer.get_ml_product(context) )
		self.domain: xa.DataArray  = trainer.target_dataset.load_global_timeslice()
		self.tcoords: DataArrayCoordinates = self.sample_target.coords
		self.icoords: DataArrayCoordinates = self.sample_input.coords
		self.time_coords: xa.DataArray = xaformat_timedeltas(self.sample_input.coords['time'])
		self.tslider: StepSlider = StepSlider('Time:', self.sample_input.sizes['time'])

		if self.prediction.ndim == 3:
			self.upsampled = to_xa( self.sample_target, trainer.get_ml_upsampled())
		else:
			coords: Dict[str,DataArrayCoordinates] = dict(time=self.tcoords['time'], channel=self.icoords['channel'], y=self.tcoords['y'], x=self.tcoords['x'])
			data: np.ndarray = trainer.get_ml_upsampled(self.context)
			self.upsampled = xa.DataArray( data, dims=['time', 'channel', 'y', 'x'], coords=coords )

		self.images_data: Dict[str,xa.DataArray] = dict( upsampled = self.upsampled, input = self.input, target = self.target, domain = self.domain )
		self.images_data[ self.result_plot_label ] = self.prediction
		self.losses: Dict[str,float] = trainer.current_losses
		self.ims = {}
		fsize = kwargs.get( 'fsize', 6.0 )
		self.tile_grid = TileSelectionGrid(self.context)
		self.ncols = (self.sample_input.shape[1]+1) if (self.sample_input is not None) else 2
		with plt.ioff():
			self.fig, self.axs = plt.subplots(nrows=2, ncols=self.ncols, figsize=[fsize*2,fsize], layout="tight")
			self.fig.canvas.mpl_connect('button_press_event', onclick)
		self.panels = [self.fig.canvas,self.tslider]
		self.tslider.set_callback( self.time_update )
		print( f"SRPlot[{self.context.name}] image types: {list(self.images_data.keys())}, losses{list(self.losses.keys())} {list(self.losses.values())}" )

	@property
	def upscale_plot_label(self) -> str:
		return "upsampled" if (self.context == LearningContext.Validation) else "domain"

	@property
	def result_plot_label(self) -> str:
		return "validation" if (self.context == LearningContext.Validation) else "prediction"

	def image(self, ir: int, ic: int) -> xa.DataArray:
		itype = self.splabels[ic][ir]
		image = self.images_data[itype]
		image.attrs['itype'] = itype
		return image

	@exception_handled
	def time_update(self, sindex: int):
		self.update_subplots(sindex)

	def plot( self ):
		self.tile_grid.overlay_grid( self.axs[1,0] )
		self.update_subplots()
		return ipw.VBox(self.panels)

	def update_subplots(self, time_index: int = 0):
		self.fig.suptitle(f'Timestep: {time_index}', fontsize=10, va="top", y=1.0)

		for irow in [0, 1]:
			for icol in [0, 1]:
				self.generate_subplot(irow, icol)

		self.fig.canvas.draw_idle()

	def generate_subplot(self, irow: int, icol: int):
		ax: Axes = self.axs[irow, icol]
		ax.set_aspect(0.5)
		image = self.get_subplot_image(irow,icol)
		vrange = cscale(image, 2.0)
		iplot: AxesImage =  image.plot.imshow(ax=ax, x="x", y="y", cmap='jet', yincrease=True, vmin=vrange[0], vmax=vrange[1])
		iplot.colorbar.remove()
		ax.set_title( self.get_subplot_title(irow,image) )
		self.ims.setdefault( (irow, icol), iplot )

	def get_subplot_title(self,irow,image) -> str:
		label = image.attrs['itype']
		rmserror = ""
		if irow == 1:
			if label in self.losses:
				rmserror = f"{self.losses[label]:.3f}" if (label in self.losses) else ""
		title = f"{label} {rmserror}"
		return title

	def get_subplot_image(self, irow: int, icol: int) -> xa.DataArray:
		image: xa.DataArray = self.image(irow, icol)
		if 'channel' in image.dims:
			image = image.isel(channel=self.channel)
		if 'time' in image.dims:
			image = image.isel(time=self.tslider.value).squeeze(drop=True)
		return image

