import math, numpy as np
import xarray as xa
from typing  import List, Tuple, Union, Optional, Dict
from fmod.base.util.ops import xaformat_timedeltas, print_data_column
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import ipywidgets as ipw
from fmod.base.util.config import cfg
from matplotlib.axes import Axes
from fmod.base.plot.widgets import StepSlider
from matplotlib.image import AxesImage
from fmod.base.util.grid import GridOps
from fmod.base.io.loader import BaseDataset
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
def mplplot( images: Dict[str,xa.DataArray] ):
	ims, pvars, ntypes, ptypes, nvars = {}, {}, len(images), [''], 1
	sample: xa.DataArray = list(images.values())[0]
	time: xa.DataArray = xaformat_timedeltas( sample.coords['time'] )
	levels: xa.DataArray = sample.coords['level']
	lunits : str = levels.attrs.get('units','')
	dayf = 24/ cfg().task.data_timestep
	lslider: ipw.IntSlider = ipw.IntSlider( value=0, min=0, max=levels.size-1, description='Level Index:', )
	tslider: ipw.IntSlider = ipw.IntSlider( value=0, min=0, max=time.size-1, description='Time Index:', )

	with plt.ioff():
		fig, axs = plt.subplots(nrows=1, ncols=ntypes, sharex=True, sharey=True, figsize=[ntypes*5, nvars*3], layout="tight")

	for itype, (tname, image) in enumerate(images.items()):
		ax = axs[ itype ]
		ax.set_aspect(0.5)
		vrange = cscale( image, 2.0 )
		tslice: xa.DataArray = image.isel(time=tslider.value)
		if "level" in tslice.dims:
			tslice = tslice.isel(level=lslider.value)
		ims[itype] =  tslice.plot.imshow( ax=ax, x="lon", y="lat", cmap='jet', yincrease=True, vmin=vrange[0], vmax=vrange[1]  )
		ax.set_title(f" {tname} ")

	@exception_handled
	def time_update(change):
		sindex = change['new']
		lindex = lslider.value
		fig.suptitle(f'Forecast day {sindex/dayf:.1f}, Level: {levels.values[lindex]:.1f} {lunits}', fontsize=10, va="top", y=1.0)
		lgm().log( f"time_update: tindex={sindex}, lindex={lindex}")
		for itype, (tname, image) in enumerate(images.items()):
			ax1 = axs[ itype ]
			tslice1: xa.DataArray =  image.isel( level=lindex, time=sindex, drop=True, missing_dims="ignore")
			ims[itype].set_data( tslice1.values )
			ax1.set_title(f"{tname}")
		fig.canvas.draw_idle()

	@exception_handled
	def level_update(change):
		lindex = change['new']
		tindex = tslider.value
		fig.suptitle(f'Forecast day {tindex/dayf:.1f}, Level: {levels.values[lindex]:.1f} {lunits}', fontsize=10, va="top", y=1.0)
		lgm().log( f"level_update: lindex={lindex}, tindex={tslider.value}")
		for itype, (tname, image) in enumerate(images.items()):
			ax1 = axs[ itype ]
			tslice1: xa.DataArray =  image.isel( level=lindex, time=tindex, drop=True, missing_dims="ignore")
			ims[itype].set_data( tslice1.values )
			ax1.set_title(f"{tname}")
		fig.canvas.draw_idle()

	tslider.observe( time_update,  names='value' )
	lslider.observe( level_update, names='value' )
	fig.suptitle(f' ** Level: {levels.values[0]:.1f} {lunits}', fontsize=10, va="top", y=1.0 )
	return ipw.VBox([tslider, lslider, fig.canvas])

