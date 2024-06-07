import math, torch, numpy as np
import xarray as xa
from datetime import datetime
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
from fmod.base.io.loader import TSet
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

class SRPlot(object):
	def __init__(self, trainer: ModelTrainer, tset: TSet, **kwargs):
		self.trainer: ModelTrainer = trainer
		self.tset: TSet = tset
		self.channel = kwargs.get('channel', 0)
		self.time_index = kwargs.get('time_index', 0)
		self.tile_index = kwargs.get('tile_index', (0, 0))
		self.splabels = [['input', self.upscale_plot_label], ['target', self.result_plot_label]]

		self.trainer.evaluate( tset, **kwargs )
		self.images_data: Dict[str, xa.DataArray] = self.update_tile_data()
		self.tslider: StepSlider = StepSlider('Time:', len(self.time_coords))
		self.losses: Dict[str,float] = trainer.current_losses
		self.ims = {}
		fsize = kwargs.get( 'fsize', 6.0 )
		self.tile_grid = TileSelectionGrid(self.tset)
		self.ncols = (self.sample_input.shape[1]+1) if (self.sample_input is not None) else 2
		with plt.ioff():
			self.fig, self.axs = plt.subplots(nrows=2, ncols=self.ncols, figsize=[fsize*2,fsize], layout="tight")
			self.fig.canvas.mpl_connect('button_press_event', self.select_point)
		self.panels = [self.fig.canvas,self.tslider]
		self.tslider.set_callback( self.time_update )

	@property
	def sample_target(self) -> xa.DataArray:
		return self.trainer.get_sample_target(self.tset)

	@property
	def tcoords(self) -> DataArrayCoordinates:
		return self.sample_target.coords

	@property
	def sample_input(self) -> xa.DataArray:
		return self.trainer.get_sample_input(self.tset)

	@property
	def icoords(self) -> DataArrayCoordinates:
		return self.sample_input.coords

	@property
	def time_coords(self) -> List[datetime]:
		return self.trainer.input_dataset(self.tset).tcoords

	def update_tile_data(self) -> Dict[str, xa.DataArray]:
		self.trainer.evaluate( self.tset, tile_index=self.tile_index, time_index=self.time_index)
		model_input: xa.DataArray = to_xa(self.sample_input, self.trainer.get_ml_input(self.tset))
		target: xa.DataArray = to_xa(self.sample_target, self.trainer.get_ml_target(self.tset))
		prediction: xa.DataArray = to_xa(self.sample_target, self.trainer.get_ml_product(self.tset))
		domain: xa.DataArray = self.trainer.target_dataset(self.tset).load_global_timeslice()

		if prediction.ndim == 3:
			upsampled = to_xa(self.sample_target, self.trainer.get_ml_upsampled(self.tset))
		else:
			coords: Dict[str, DataArrayCoordinates] = dict(time=self.tcoords['time'], channel=self.icoords['channel'], y=self.tcoords['y'], x=self.tcoords['x'])
			data: np.ndarray = self.trainer.get_ml_upsampled(self.tset)
			upsampled = xa.DataArray(data, dims=['time', 'channel', 'y', 'x'], coords=coords)

		images_data: Dict[str, xa.DataArray] = dict(upsampled=upsampled, input=model_input, target=target, domain=domain)
		images_data[self.result_plot_label] = prediction
		lgm().log(f"\n update_tile_data ---> images = {list(images_data.keys())}")
		return images_data

	def select_point(self,event):
		lgm().log(f'Mouse click: button={event.button}, dbl={event.dblclick}, x={event.xdata:.2f}, y={event.ydata:.2f}')
		selected_tile: Optional[Tuple[int, int]] = self.tile_grid.get_selected(event.xdata, event.ydata)
		self.select_tile( selected_tile )

	def select_tile(self, selected_tile: Tuple[int,int]):
		print(f" ---> selected_tile: {selected_tile}")
		if selected_tile is not None:
			self.tile_index = selected_tile
			self.update_tile_data()
			self.update_subplots()
			lgm().log( f" ---> selected_tile = {selected_tile}")

	@property
	def upscale_plot_label(self) -> str:
		return "upsampled" # if (self.tset == TSet.Validation) else "domain"

	@property
	def result_plot_label(self) -> str:
		return "validation" if (self.tset == TSet.Validation) else "prediction"

	def image(self, ir: int, ic: int) -> xa.DataArray:
		itype = self.splabels[ic][ir]
		image = self.images_data[itype]
		image.attrs['itype'] = itype
		return image

	@exception_handled
	def time_update(self, sindex: int):
		lgm().log(f"\n time_update ---> sindex = {sindex}")
		self.time_index = sindex
		self.update_tile_data()
		self.update_subplots()

	def plot( self ):
		self.tile_grid.overlay_grid( self.axs[1,0] )
		self.update_subplots()
		return ipw.VBox(self.panels)

	@property
	def display_time(self) -> str:
		ctime: datetime = self.time_coords[self.time_index]
		return ctime.strftime("%m/%d/%Y:%H")

	def update_subplots(self):
		self.fig.suptitle(f'Time: {self.display_time}', fontsize=10, va="top", y=1.0)

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
		print( f"\n generate_subplot({irow},{icol}): shape={iplot.get_shape()}")

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
			batch_time_index = self.time_index % self.trainer.input_dataset(self.tset).steps_per_batch
			lgm().log( f"get_subplot_image: time_index={self.time_index}, batch_time_index={batch_time_index} --> image{image.dims}{list(image.shape)}",display=True)
			image = image.isel(time=batch_time_index).squeeze(drop=True)
		return image

