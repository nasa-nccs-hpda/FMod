import torch, numpy as np
import xarray as xa
from typing  import List, Tuple, Optional, Dict
from fmod.base.io.loader import TSet, batchDomain
from fmod.base.util.config import cfg
from fmod.base.util.array import array2tensor, downsample, upsample, xa_downsample, xa_upsample
import ipywidgets as ipw
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from xarray.core.coordinates import DataArrayCoordinates
from fmod.controller.dual_trainer import ModelTrainer
from fmod.base.io.loader import TSet, srRes
from fmod.view.tile_selection_grid import TileSelectionGrid
from fmod.view.plot.widgets import StepSlider
from fmod.base.util.logging import lgm, exception_handled
from fmod.view.plot.base import Plot

colors = ["red", "blue", "green", "cyan", "magenta", "yellow", "grey", "brown", "pink", "purple", "orange", "black"]

def flex(weight: int) -> ipw.Layout:
	return ipw.Layout(flex=f'1 {weight} auto', width='auto')
def rms( dvar: xa.DataArray, **kwargs ) -> float:
	varray: np.ndarray = dvar.isel( **kwargs, missing_dims="ignore", drop=True ).values
	return np.sqrt( np.mean( np.square( varray ) ) )

def tensor( dvar: xa.DataArray ) -> torch.Tensor:
	return torch.from_numpy( dvar.values.squeeze() )

def rmse( diff: xa.DataArray, **kw ) -> xa.DataArray:
	rms_error = np.array( [ rms(diff, tiles=iT, **kw) for iT in range(diff.shape[0]) ] )
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

class ResultImagePlot(Plot):
	def __init__(self, trainer: ModelTrainer, tset: TSet, **kwargs):
		super(ResultImagePlot, self).__init__(trainer, **kwargs)
		self.tset: TSet = tset
		self.time_index: int = kwargs.get( 'time_id', 0 )
		self.losses = None
		self.tileId: int = kwargs.get( 'tile_id', 0 )
		self.varId: int = kwargs.get( 'var_id', 0 )
		self.images_data: Dict[str, xa.DataArray] = self.update_tile_data(update_model=True)
		self.tslider: StepSlider = StepSlider('Time:', self.time_index, len(self.trainer.data_timestamps[tset]) )
		self.plot_titles: List[str] = list(self.images_data.keys())
		self.ims = {}
		self.callbacks = dict(button_press_event=self.select_point)
		self.create_figure( nrows=4, ncols=1, callbacks=self.callbacks, title='SRes Loss Over Training Epochs' )
		self.tslider.set_callback( self.time_update )

	@property
	def sample_target(self) -> xa.DataArray:
		return self.trainer.get_sample_target()

	@property
	def tcoords(self) -> DataArrayCoordinates:
		return self.sample_target.coords

	@property
	def sample_input(self) -> xa.DataArray:
		return self.trainer.get_sample_input()

	@property
	def icoords(self) -> DataArrayCoordinates:
		return self.sample_input.coords

	@property
	def batch_domain(self) -> batchDomain:
		return self.trainer.batch_domain

	def update_tile_data( self, **kwargs ) -> Dict[str, xa.DataArray]:
		images_data, eval_losses = self.trainer.process_image( self.tset, self.varId, self.time_index, interp_loss=True, **kwargs )
		if len( eval_losses ) > 0:
			self.losses = eval_losses
			lgm().log(f"update_tile_data ---> images = {list(images_data.keys())}")
			return images_data

	def select_point(self,event):
		lgm().log(f'Mouse click: button={event.button}, dbl={event.dblclick}, x={event.xdata:.2f}, y={event.ydata:.2f}')

	@property
	def upscale_plot_label(self) -> str:
		return "interpolated"

	@property
	def result_plot_label(self) -> str:
		return "model"

	@exception_handled
	def time_update(self, sindex: int):
		lgm().log(f"\n time_update ---> sindex = {sindex}")
		self.time_index = sindex
		self.images_data = self.update_tile_data()
		self.update_subplots()


	def plot( self ) -> ipw.Box:
		self.update_subplots()
		panels = [self.fig.canvas, self.tslider]
		return ipw.VBox(panels)

	@property
	def display_time(self) -> str:
		return str(self.time_index)
	#	ctime: datetime = self.time_coords[self.time_index]
	#	return ctime.strftime("%m/%d/%Y:%H")

	def update_subplots(self):
		self.fig.suptitle(f'Time: {self.display_time}', fontsize=10, va="top", y=1.0)
		for iplot in range(4):
				self.generate_subplot(iplot)
		self.fig.canvas.draw_idle()

	#	cbar.clim(vmin, vmax)

	def generate_subplot(self, iplot: int):
		ax: Axes = self.axs[iplot]
		ax.set_aspect(0.5)
		ax.set_xlim([0, 100])
		ax.set_ylim([0, 100])
		ptype: str = self.plot_titles[iplot]
		image: xa.DataArray = self.images_data[ptype]
		vrange = [np.nanmin(image.values), np.nanmax(image.values)]
		print( f"subplot_image[{ptype}]: image{image.dims}{image.shape}, vrange={vrange}")
		iplot: AxesImage =  image.plot.imshow(ax=ax, x="x", y="y", cmap='jet', yincrease=True, add_colorbar=True ) #, vmin=vrange[0], vmax=vrange[1] )
		ax.set_title( self.get_subplot_title(ptype) )
		self.ims[ iplot ] = iplot

	def get_subplot_title(self, ptype: str) -> str:
		loss: float = None
		if   ptype == "interp": loss = self.losses.get("interp",0.0)
		elif ptype == "output": loss = self.losses.get('model', 0.0)
		return ptype if (loss is None) else f"{ptype}, loss={loss*1000:.3f}"



