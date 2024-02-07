import math, numpy as np
import xarray as xa
from typing  import List, Tuple, Union, Optional, Dict
from fmod.base.util.ops import xaformat_timedeltas, print_data_column
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import ipywidgets as ipw
from torch import Tensor
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from fmod.base.util.grid import GridOps
from fmod.base.io.loader import BaseDataset
from fmod.base.util.logging import lgm, exception_handled, log_timing

colors = ["red", "blue", "green", "cyan", "magenta", "yellow", "grey", "brown", "pink", "purple", "orange", "black"]

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
	tensor_roles = ["target", "prediction"]

	def __init__(self, dataset: BaseDataset, targets: List[Tensor], prediction: List[Tensor], **kwargs ):
		figsize = kwargs.pop('figsize',[10, 5])
		(nchan, nlat, nlon) = targets[0].shape[-3:]
		self.dataset: BaseDataset = dataset
		self.chanids: List[str] = self.dataset.chanIds['target']
		with plt.ioff():
			fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=figsize, layout="tight", **kwargs)
		self.fig: plt.Figure = fig
		self.axs: Axes = axs
		for ax in axs.flat: ax.set_aspect(0.5)
		self.ichannel: int = 0
		self.istep: int = 0
		self.gridops = GridOps(nlat, nlon)
		self.plot_data: Tuple[List[Tensor],List[Tensor]] = ( targets, prediction )
		self.cslider: ipw.IntSlider = ipw.IntSlider(value=0, min=0, max=nchan-1, description='Channel Index:', )
		self.sslider: ipw.IntSlider = ipw.IntSlider(value=0, min=0, max=len(targets)-1, description='Step Index:', )
		self.vrange: Tuple[float,float] = (0.0,0.0)
		self.ims: List[Optional[AxesImage]] = [None,None,None]
		self.cslider.observe(self.channel_update, names='value')
		self.sslider.observe(self.step_update, names='value')
		self.format_plot()

	def format_plot(self):
		self.fig.suptitle(f'step index={self.istep}, channel index={self.ichannel}', fontsize=10, va="top", y=1.0)

	@exception_handled
	def plot(self, **kwargs):
		cmap = kwargs.pop('cmap', 'jet')
		origin = kwargs.pop('origin', 'lower' )
		for ip, pdata in enumerate(self.plot_data):
			ax = self.axs[ip]
			ax.set_title(f"{self.tensor_roles[ip]}")
			image_data: np.ndarray = self.image_data( ip, pdata[self.istep] )
			plot_args = dict( cmap=cmap, origin=origin, vmin=self.vrange[0], vmax=self.vrange[1], **kwargs )
			self.ims[ip] = ax.imshow( image_data, **plot_args)
		return ipw.VBox([self.cslider, self.sslider, self.fig.canvas])

	@exception_handled
	def step_update(self, change):
		self.istep = change['new']
		lgm().log(f"Step update: istep={self.istep}, ichannel={self.ichannel}")
		self.refresh()

	@exception_handled
	def channel_update(self, change):
		self.ichannel = change['new']
		lgm().log( f"Channel update: istep={self.istep}, ichannel={self.ichannel}")
		self.fig.suptitle(self.channel_title, fontsize=12)
		self.refresh()

	@property
	def channel_title(self) -> str:
		return self.chanids[self.ichannel]

	@exception_handled
	def refresh(self):
		for ip, pdata in enumerate(self.plot_data):
			self.ims[ip].set_data( self.image_data( ip, pdata[self.istep] ) )
		self.format_plot()
		self.fig.canvas.draw_idle()

	def image_data(self, ip: int, timeslice: Tensor) -> np.ndarray:
		image_data: Tensor = timeslice[0, self.ichannel] if (timeslice.dim() == 4) else timeslice[self.ichannel]
		if ip == 0: self.vrange = self.gridops.color_range(image_data, 2.0)
		return image_data.cpu().numpy()




