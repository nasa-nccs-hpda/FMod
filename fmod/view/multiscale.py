import math, numpy as np
import xarray as xa
from torch import nn
from typing  import List, Tuple, Union, Optional, Dict
from fmod.base.util.ops import xaformat_timedeltas, print_data_column
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import ipywidgets as ipw
from fmod.base.util.config import cfg
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from fmod.base.util.grid import GridOps
from fmod.base.io.loader import BaseDataset
from fmod.base.plot.widgets import StepSlider
from fmod.base.util.logging import lgm, exception_handled, log_timing

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
def mplplot( images: Dict[str,xa.DataArray], **kwargs ):
	ims, pvars, ntypes, ptypes, nvars = {}, {}, len(images), [''], 1
	sample: xa.DataArray = list(images.values())[0]
	time: xa.DataArray = xaformat_timedeltas( sample.coords['time'] )
	channels: List[str] = sample.coords['channel'].values.tolist()
	cslider: StepSlider = StepSlider( 'Channel:', len(channels)  )
	tslider: StepSlider = StepSlider( 'Time:', time.size  )
	fsize = kwargs.get( 'fsize', 5.0 )


	with plt.ioff():
		fig, axs = plt.subplots(nrows=1, ncols=ntypes, sharex=True, sharey=True, figsize=[ntypes*fsize*1.4, nvars*fsize], layout="tight")

	for itype, (tname, image) in enumerate(images.items()):
		ax = axs[ itype ]
		ax.set_aspect(0.5)
		vrange = cscale( image, 2.0 )
		tslice: xa.DataArray = image.isel(time=tslider.value)
		cslice: xa.DataArray = tslice.isel(channel=cslider.value).fillna( 0.0 )
		ims[itype] =  cslice.plot.imshow( ax=ax, x="i", y="j", cmap='jet', yincrease=True, vmin=vrange[0], vmax=vrange[1]  )
		ax.set_title(f" {tname} ")

	@exception_handled
	def time_update(sindex: int):
		cindex = cslider.value
		fig.suptitle(f'Timestep: {sindex}, Channel: {channels[cindex]}', fontsize=10, va="top", y=1.0)
		lgm().log( f"time_update: tindex={sindex}, cindex={cindex}")
		for itype, (tname, image) in enumerate(images.items()):
			ax1 = axs[ itype ]
			tslice1: xa.DataArray =  image.isel( channels=cindex, time=sindex, drop=True, missing_dims="ignore").fillna( 0.0 )
			ims[itype].set_data( tslice1.values )
			ax1.set_title(f"{tname}")
		fig.canvas.draw_idle()

	@exception_handled
	def channel_update(cindex: int):
		sindex = tslider.value
		fig.suptitle(f'Forecast day {sindex}, Channel: {channels[cindex]}', fontsize=10, va="top", y=1.0)
		lgm().log( f"level_update: cindex={cindex}, tindex={tslider.value}")
		for itype, (tname, image) in enumerate(images.items()):
			ax1 = axs[ itype ]
			tslice1: xa.DataArray =  image.isel( channels=cindex, time=sindex, drop=True, missing_dims="ignore").fillna( 0.0 )
			ims[itype].set_data( tslice1.values )
			ax1.set_title(f"{tname}")
		fig.canvas.draw_idle()

	tslider.set_callback( time_update )
	cslider.set_callback( channel_update )
	fig.suptitle(f' ** Channel: {channels[0]}', fontsize=10, va="top", y=1.0 )
	return ipw.VBox([tslider, cslider, fig.canvas])

