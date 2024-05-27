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

def tensor( dvar: xa.DataArray ) -> torch.Tensor:
	return torch.from_numpy( dvar.values )

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


def create_plot_data( inputs: np.ndarray, targets: np.ndarray, predictions: np.ndarray, upsampled: np.ndarray, sample_input: xa.DataArray, sample_target: xa.DataArray ) -> Dict[str,xa.DataArray]:

	print(f"sample_input shape = {sample_input.shape}")
	print(f"sample_target shape = {sample_target.shape}")
	print( f"inputs shape = {inputs.shape}")
	print(f"targets shape = {targets.shape}")
	print(f"predictions shape = {predictions.shape}")
	print(f"upsampled shape = {upsampled.shape}")
	tc, ic = sample_target.coords, sample_input.coords
	if predictions.ndim == 3: x_upsampled = sample_target.copy( data=upsampled.reshape(sample_target.shape))
	else:                     x_upsampled = xa.DataArray(upsampled, dims=['time', 'channel', 'y', 'x'], coords=dict(time=tc['time'], channel=ic['channel'], y=tc['y'], x=tc['x']))

	return dict(    input=       sample_input.copy(  data=inputs.reshape(sample_input.shape) ),
					targets=     sample_target.copy( data=targets.reshape(sample_target.shape) ),
					predictions= sample_target.copy( data=predictions.reshape(sample_target.shape) ),
					upsampled=   x_upsampled )

@exception_handled
def mplplot( images: Dict[str,xa.DataArray], **kwargs ):
	ims, labels, rms_errors, target = {}, {}, {}, ""
	sample: xa.DataArray = images['input']
	print( f"Plotting {len(images)} images, sample{sample.dims}: {sample.shape}")
	batch: xa.DataArray = xaformat_timedeltas( sample.coords['time'] )
	tslider: StepSlider = StepSlider( 'Time:', batch.size  )
	fsize = kwargs.get( 'fsize', 6.0 )
	ncols = sample.shape[1]+1

	with plt.ioff():
		fig, axs = plt.subplots(nrows=2, ncols=ncols, figsize=[fsize*2,fsize], layout="tight")

	for irow in [0,1]:
		for icol in range(ncols):
			ax = axs[ irow, icol ]
			rmserror = ""
			if icol == ncols-1:
				labels[(irow,icol)] = ['targets','predictions'][irow]
				image = images[ labels[(irow,icol)] ]
				if irow == 0: target = image
			else:
				labels[(irow,icol)] = ['input', 'upsampled'][irow]
				image = images[ labels[(irow,icol)] ]
				image = image.isel( channel=icol )
			ax.set_aspect(0.5)
			vrange = cscale( image, 2.0 )
			tslice: xa.DataArray = image.isel(time=tslider.value).squeeze(drop=True)
			ims[(irow,icol)] = tslice.plot.imshow( ax=ax, x="x", y="y", cmap='jet', yincrease=True, vmin=vrange[0], vmax=vrange[1]  )
			if irow == 1: rmserror = f"{l2loss(tensor(image),tensor(target)):.3f}"
			ax.set_title(f" {labels[(irow,icol)]} {rmserror}")

	@exception_handled
	def time_update(sindex: int):
		fig.suptitle(f'Timestep: {sindex}', fontsize=10, va="top", y=1.0)
		lgm().log( f"time_update: tindex={sindex}")
		target = None
		for irow in [0, 1]:
			for icol in range(ncols):
				ax1 = axs[ irow, icol ]
				rmserror = ""
				if icol == ncols - 1:
					labels[(irow, icol)] = ['targets', 'predictions'][irow]
					image = images[labels[(irow, icol)]]
					if irow == 0: target = image
				else:
					labels[(irow, icol)] = ['input', 'upsampled'][irow]
					image = images[labels[(irow, icol)]]
					image = image.isel(channel=icol)
				tslice1: xa.DataArray =  image.isel( time=sindex, drop=True, missing_dims="ignore").fillna( 0.0 )
				ims[(irow,icol)].set_data( tslice1.values.squeeze() )
				if irow == 1: rmserror = f"{RMSE(tslice1 - target):.3f}"
				ax1.set_title(f"{labels[(irow,icol)]} {rmserror}")
		fig.canvas.draw_idle()


	tslider.set_callback( time_update )
	fig.suptitle(f' ** ', fontsize=10, va="top", y=1.0 )
	return ipw.VBox([fig.canvas,tslider])

