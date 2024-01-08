import math, numpy as np
import xarray as xa
from typing  import List, Tuple, Union, Optional, Dict
from fmod.base.util.ops import xaformat_timedeltas, print_data_column
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import ipywidgets as ipw
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
def mplplot( target: xa.Dataset, vnames: List[str],  task_spec: Dict, **kwargs ):
	ims, pvars, nvars, ptypes = {}, {}, len(vnames), ['']
	forecast: Optional[xa.Dataset] = kwargs.pop('forecast',None)
	time: xa.DataArray = xaformat_timedeltas( target.coords['time'] )
	levels: xa.DataArray = target.coords['level']
	lunits : str = levels.attrs.get('units','')
	dayf = 24/task_spec['data_timestep']
	target.assign_coords( time=time )
	if forecast is not None:
		forecast.assign_coords(time=time)
		ptypes = ['target', 'forecast', 'difference']
	ncols =  len( ptypes )
	lslider: ipw.IntSlider = ipw.IntSlider( value=0, min=0, max=levels.size-1, description='Level Index:', )
	tslider: ipw.IntSlider = ipw.IntSlider( value=0, min=0, max=time.size-1, description='Time Index:', )
	print_data_column( target, vnames[0], **kwargs )
	errors: Dict[str,xa.DataArray] = {}

	with plt.ioff():
		fig, axs = plt.subplots(nrows=nvars, ncols=ncols, sharex=True, sharey=True, figsize=[ncols*5, nvars*3], layout="tight")

	for iv, vname in enumerate(vnames):
		tvar: xa.DataArray = normalize(target,vname,**kwargs)
		plotvars = [ tvar ]
		if forecast is not None:
			fvar: xa.DataArray = normalize(forecast,vname,**kwargs)
			diff: xa.DataArray = tvar - fvar
			errors[vname] = rmse(diff)
			plotvars = plotvars + [ fvar, diff ]
		vrange = None
		for it, pvar in enumerate( plotvars ):
			ax = axs[ iv ] if ncols == 1 else axs[ iv, it ]
			ax.set_aspect(0.5)
			if it != 1: vrange = cscale( pvar, 2.0 )
			tslice: xa.DataArray = pvar.isel(time=tslider.value)
			if "level" in tslice.dims:
				tslice = tslice.isel(level=lslider.value)
			ims[(iv,it)] =  tslice.plot.imshow( ax=ax, x="lon", y="lat", cmap='jet', yincrease=True, vmin=vrange[0], vmax=vrange[1]  )
			pvars[(iv,it)] =  pvar
			ax.set_title(f"{vname} {ptypes[it]}")

	@exception_handled
	def time_update(change):
		sindex = change['new']
		lindex = lslider.value
		fig.suptitle(f'Forecast day {sindex/dayf:.1f}, Level: {levels.values[lindex]:.1f} {lunits}', fontsize=10, va="top", y=1.0)
		lgm().log( f"time_update: tindex={sindex}, lindex={lindex}")
		for iv1, vname1 in enumerate(vnames):
			for it1 in range(ncols):
				ax1 = axs[ iv ] if ncols == 1 else axs[ iv, it ]
				im1, dvar1 = ims[ (iv1, it1) ], pvars[ (iv1, it1) ]
				tslice1: xa.DataArray =  dvar1.isel( level=lindex, time=sindex, drop=True, missing_dims="ignore")
				im1.set_data( tslice1.values )
				ax1.set_title(f"{vname1} {ptypes[it1]}")
				lgm().log(f" >> Time-update {vname1} {ptypes[it1]}: level={lindex}, time={sindex}, shape={tslice1.shape}")
		fig.canvas.draw_idle()

	@exception_handled
	def level_update(change):
		lindex = change['new']
		tindex = tslider.value
		fig.suptitle(f'Forecast day {tindex/dayf:.1f}, Level: {levels.values[lindex]:.1f} {lunits}', fontsize=10, va="top", y=1.0)
		lgm().log( f"level_update: lindex={lindex}, tindex={tslider.value}")
		for iv1, vname1 in enumerate(vnames):
			for it1 in range(ncols):
				ax1 = axs[ iv ] if ncols == 1 else axs[ iv, it ]
				im1, dvar1 = ims[ (iv1, it1) ], pvars[ (iv1, it1) ]
				tslice1: xa.DataArray =  dvar1.isel( level=lindex,time=tindex, drop=True, missing_dims="ignore")
				im1.set_data( tslice1.values )
				ax1.set_title(f"{vname1} {ptypes[it1]}")
				lgm().log(f" >> Level-update {vname1} {ptypes[it1]}: level={lindex}, time={tindex}, mean={tslice1.values.mean():.4f}, std={tslice1.values.std():.4f}")
		fig.canvas.draw_idle()

	tslider.observe( time_update,  names='value' )
	lslider.observe( level_update, names='value' )
	fig.suptitle(f' ** Forecast day 0, Level: {levels.values[0]:.1f} {lunits}', fontsize=10, va="top", y=1.0 )
	return ipw.VBox([tslider, lslider, fig.canvas])

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