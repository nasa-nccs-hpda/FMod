import math, numpy as np
import xarray as xa
from typing  import List, Tuple, Optional, Dict
from fmod.base.util.ops import xaformat_timedeltas
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import ipywidgets as ipw
from view.plot import StepSlider
from matplotlib.image import AxesImage
from fmod.base.util.grid import GridOps
from fmod.base.util.logging import lgm, exception_handled
from view.plot import color_range

colors = ["red", "blue", "green", "cyan", "magenta", "yellow", "grey", "brown", "pink", "purple", "orange", "black"]

def flex(weight: int) -> ipw.Layout:
	return ipw.Layout(flex=f'1 {weight} auto', width='auto')
def rms( dvar: xa.DataArray, **kwargs ) -> float:
	varray: np.ndarray = dvar.isel( **kwargs, missing_dims="ignore", drop=True ).values
	return np.sqrt( np.mean( np.square( varray ) ) )

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

@exception_handled
def mplplot_error( target: xa.Dataset, forecast: xa.Dataset, vnames: List[str],  **kwargs ):
	ftime: np.ndarray = xaformat_timedeltas( target.coords['time'], form="day", strf=False ).values
	tvals = list( range( 1, round(float(ftime[-1]))+1, math.ceil(ftime.size/10) ) )
	with plt.ioff():
		fig, ax = plt.subplots(nrows=1, ncols=1,  figsize=[ 9, 6 ], layout="tight")

	for iv, vname in enumerate(vnames):
		tvar: xa.DataArray = normalize(target,vname,**kwargs)
		fvar: xa.DataArray = normalize(forecast,vname,**kwargs)
		error: xa.DataArray = rmse(tvar-fvar).assign_coords(time=ftime).rename( time = "time (days)")
		error.plot.line( ax=ax, color=colors[iv], label=vname )

	ax.set_title(f"  Forecast Error  ")
	ax.xaxis.set_major_locator(ticker.FixedLocator(tvals))
	ax.legend()
	return fig.canvas

class ResultsPlotter:
	tensor_roles = [ "target", "prediction", "interpolated" ]

	def __init__(self, inputs: List[xa.DataArray], targets: List[xa.DataArray], predictions: List[xa.DataArray], interpolates: Optional[List[xa.DataArray]] = None, **kwargs ):
		(self.nchan, nlat, nlon) = targets[0].shape[-3:]
		(self.fig, self.axs) = (None,None)
		self.inputs: List[xa.DataArray] = inputs
		self.targets: List[xa.DataArray] = targets
		self.predictions: List[xa.DataArray] = predictions
		self.interpolates: Optional[List[xa.DataArray]] = interpolates
		self.nsteps = len(targets)
		self.chanids: List[str] = kwargs.pop('chanids',[])
		self.ichannel: int = 0
		self.istep: int = 0
		self.create_figure(**kwargs)
		self.gridops = GridOps(nlat, nlon, device)
		self.cslider: StepSlider = StepSlider( 'Channel:', self.nchan,  self.channel_update )
		self.sslider: StepSlider = StepSlider( 'Step:', self.nsteps,  self.step_update )
		self.vrange: Tuple[float,float] = (0.0,0.0)
		self.ims: List[Optional[AxesImage]] = [None,None,None]
		self.input_images = [self.targets, self.predictions]
		if self.interpolates is not None: self.input_images.append(self.interpolates)
		self.format_plot()

	def format_plot(self):
		self.fig.suptitle( self.channel_title, fontsize=10, va="top", y=1.0 )

	def create_figure(self, **kwargs ):
		figsize = kwargs.pop('figsize',(12, 5))
		with plt.ioff():
			with plt.ioff():
				self.fig, self.axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=figsize, layout="tight", **kwargs)
		for ax in self.axs.flat:
			ax.set_aspect(0.5)
			ax.set_axis_off()
		print(self.fig.__class__)

	@exception_handled
	def plot(self, **kwargs):
		cmap = kwargs.pop('cmap', 'jet')
		origin = kwargs.pop('origin', 'lower' )
		for ip, image_arrays in enumerate( self.input_images ):
			ax = self.axs[ip]
			ax.set_title(f"{self.tensor_roles[ip]}")
			image_data: np.ndarray = self.image_data( ip, image_arrays[self.istep].values )
			plot_args = dict( cmap=cmap, origin=origin, vmin=self.vrange[0], vmax=self.vrange[1], **kwargs )
			lgm().log(f" ** image{ip}: shape={image_data.shape}: args={plot_args}",display=True)
			self.ims[ip] = ax.imshow( image_data, **plot_args)
		return ipw.VBox( [self.fig.canvas, self.cslider, self.sslider] )

	@exception_handled
	def step_update(self, istep: int ):
		self.istep = istep
		lgm().log(f"Step update: istep={self.istep}, ichannel={self.ichannel}")
		self.refresh()

	@exception_handled
	def channel_update(self, ichannel: int ):
		self.ichannel = ichannel
		lgm().log( f"Channel update: istep={self.istep}, ichannel={self.ichannel}, title={self.channel_title}")
		self.refresh()

	@property
	def channel_title(self) -> str:
		return f"{self.chanids[self.ichannel]}:  step={self.istep}"

	@exception_handled
	def refresh(self):
		for ip, image_arrays in enumerate( self.input_images ):
			self.ims[ip].set_data( self.image_data( ip, image_arrays[self.istep].values ) )
		self.format_plot()
		self.fig.canvas.draw_idle()

	def image_data(self, ip: int, timeslice: np.ndarray) -> np.ndarray:
		image_data: np.ndarray = timeslice[0, self.ichannel] if (timeslice.ndim == 4) else timeslice[self.ichannel]
		if ip == 0: self.vrange = color_range(image_data, 2.0)
		return image_data




