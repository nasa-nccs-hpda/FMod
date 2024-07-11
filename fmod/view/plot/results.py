import torch, numpy as np
import xarray as xa
from typing  import List, Tuple, Optional, Dict
from fmod.base.util.array import array2tensor, downsample, upsample
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

class ResultPlot(Plot):
	def __init__(self, trainer: ModelTrainer, tset: TSet, **kwargs):
		super(ResultPlot, self).__init__(trainer, **kwargs)
		self.tset: TSet = tset
		self.tile_grid: TileSelectionGrid = TileSelectionGrid(self.tset)
		self.tile_grid.create_tile_recs(**kwargs)
		self.channelx: int = kwargs.get('channel', 0)
		self.time_index: int = kwargs.get( 'time_id', 0 )
		self.tileId: int = kwargs.get( 'tile_id', 0 )
		self.channel: int = kwargs.get( 'channel', 0 )
		self.tile_index: Tuple[int,int] = self.tile_grid.get_tile_coords( self.tileId )
		self.splabels = [['input', self.upscale_plot_label], ['target', self.result_plot_label]]
		self.losses: Dict[str,float] = {}
		self.images_data: Dict[str, xa.DataArray] = self.update_tile_data(update_model=True)
		self.tslider: StepSlider = StepSlider('Time:', self.time_index, self.sample_input.sizes['time'] )
		self.sslider: StepSlider = StepSlider('Tile:', self.tileId, self.tile_grid.ntiles )
		self.plot_titles: List[List[str]] = [ ['input', 'target'], ['upsample', 'model'] ]
		self.ims = {}
		self.ncols = (self.sample_input.shape[1]+1) if (self.sample_input is not None) else 2
		self.callbacks = dict(button_press_event=self.select_point)
		self.create_figure( nrows=2, ncols=self.ncols, sharex=True, sharey=True, callbacks=self.callbacks, title='SRes Loss Over Training Epochs' )
		self.panels = [ self.fig.canvas, self.tslider, self.sslider ]
		self.tslider.set_callback( self.time_update )
		self.sslider.set_callback( self.tile_update )

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

	def update_tile_data( self, **kwargs ) -> Dict[str, xa.DataArray]:
		self.tile_index = self.tile_grid.get_tile_coords( self.tileId )
		self.losses = self.trainer.evaluate( self.tset, tile_index=self.tile_index, time_index=self.time_index, upsample=True, **kwargs )
		input_data = self.trainer.get_ml_input(self.tset)
		target_data = self.trainer.get_ml_target(self.tset)
		product_data =  self.trainer.get_ml_product(self.tset)
		print( f"input_data shape = {input_data.shape}")
		print( f"target_data shape = {target_data.shape}")
		print(f"product_data shape = {product_data.shape}")
		model_input: xa.DataArray = xa.DataArray( input_data, dims=['batch','channels','y','x'] )
		target: xa.DataArray = xa.DataArray( target_data, dims=['batch','channels','y','x'] )
		prediction: xa.DataArray = xa.DataArray( product_data, dims=['batch','channels','y','x'] )
		domain: xa.DataArray = self.trainer.target_dataset(self.tset).load_global_timeslice(index=0)
		lgm().log( f"update_tile_data{self.tile_index}: prediction shape = {prediction.shape}, target shape = {target.shape}")

		if prediction.ndim == 3:
			upsampled = to_xa(self.sample_target, self.trainer.get_ml_upsampled(self.tset))
		else:
			# coords: Dict[str, DataArrayCoordinates] = dict(time=self.tcoords['time'], channels=self.icoords['channel'], y=self.tcoords['y'], x=self.tcoords['x'])
			data: np.ndarray = upsample( self.trainer.input[self.tset] ).numpy()
			upsampled = xa.DataArray(data, dims=['time', 'channels', 'y', 'x'] ) # , coords=coords)

		images_data: Dict[str, xa.DataArray] = dict(upsample=upsampled, input=model_input, target=target, domain=domain)
		images_data[self.result_plot_label] = prediction
		lgm().log(f"update_tile_data ---> images = {list(images_data.keys())}")
		return images_data

	def select_point(self,event):
		lgm().log(f'Mouse click: button={event.button}, dbl={event.dblclick}, x={event.xdata:.2f}, y={event.ydata:.2f}')
		selected_tile: Optional[Tuple[int, int]] = self.tile_grid.get_selected(event.xdata, event.ydata)
		self.select_tile( selected_tile )

	def select_tile(self, selected_tile: Tuple[int,int]):
		print(f" ---> selected_tile: {selected_tile}")
		if selected_tile is not None:
			self.tile_index = selected_tile
			self.images_data = self.update_tile_data()
			self.update_subplots()
			lgm().log( f" ---> selected_tile = {selected_tile}")

	@property
	def upscale_plot_label(self) -> str:
		return "upsample"

	@property
	def result_plot_label(self) -> str:
		return "model"

	def image(self, ir: int, ic: int) -> xa.DataArray:
		itype = self.splabels[ic][ir]
		image = self.images_data[itype]
		image.attrs['itype'] = itype
		return image

	@exception_handled
	def time_update(self, sindex: int):
		lgm().log(f"\n time_update ---> sindex = {sindex}")
		self.time_index = sindex
		self.images_data = self.update_tile_data()
		self.update_subplots()

	@exception_handled
	def tile_update(self, sindex: int):
		lgm().log( f"\n tile_update ---> sindex = {sindex}" )
		self.tileId = sindex
		self.images_data = self.update_tile_data()
		self.update_subplots()


	def plot( self ) -> ipw.Box:
		# self.tile_grid.overlay_grid( self.axs[1,0] )
		self.update_subplots()
		return ipw.VBox(self.panels)

	@property
	def display_time(self) -> str:
		return str(self.time_index)
	#	ctime: datetime = self.time_coords[self.time_index]
	#	return ctime.strftime("%m/%d/%Y:%H")

	def update_subplots(self):
		self.fig.suptitle(f'Time: {self.display_time}, Tile: {self.tile_index}', fontsize=10, va="top", y=1.0)

		for irow in [0, 1]:
			for icol in [0, 1]:
				self.generate_subplot(irow, icol)

		self.fig.canvas.draw_idle()

	def generate_subplot(self, irow: int, icol: int):
		ax: Axes = self.axs[irow, icol]
		ax.set_aspect(0.5)
		ts: Dict[str, int] = self.tile_grid.tile_grid.get_full_tile_size()
		ax.set_xlim([0, ts['x']])
		ax.set_ylim([0, ts['y']])
		image: xa.DataArray = self.get_subplot_image(irow, icol, ts)

		vrange = cscale(image, 2.0)
		iplot: AxesImage =  image.plot.imshow(ax=ax, x="x", y="y", cmap='jet', yincrease=True, vmin=vrange[0], vmax=vrange[1])
		iplot.colorbar.remove()
		ax.set_title( self.get_subplot_title(irow,icol) )
		self.ims[ (irow, icol) ] = iplot

	def get_subplot_title(self,irow,icol) -> str:
		label = self.plot_titles[irow][icol]
		rmserror = ""
		if irow == 1:
			loss: float = self.losses[label]
			rmserror = f"{loss*1000:.3f}"
		title = f"{label} {rmserror}"
		return title

	def get_subplot_image(self, irow: int, icol: int, ts: Dict[str, int] ) -> xa.DataArray:
		image: xa.DataArray = self.image(irow, icol)
		if 'channel' in image.dims:
			image = image.isel(channels=self.channel)
		if 'time' in image.dims:
			batch_time_index = self.time_index % self.trainer.input_dataset(self.tset).batch_size
			# lgm().log( f"get_subplot_image: time_index={self.time_index}, batch_time_index={batch_time_index} --> image{image.dims}{list(image.shape)}")
			image = image.isel(time=batch_time_index).squeeze(drop=True)
		dx, dy = ts['x']/image.shape[1], ts['y']/image.shape[0]
		image = image.assign_coords( x=np.linspace(-dx/2, ts['x']+dx/2, image.shape[1] ), y=np.linspace(-dy/2, ts['y']+dy/2, image.shape[0] ) )
		return image

