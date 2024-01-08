from netCDF4 import Dataset, Variable
import numpy as np
import xarray as xa

def nc4_write_array( path: str, var: xa.DataArray ):
	ncfile = Dataset( path, mode='w', format='NETCDF4')
	for idim, dname in enumerate(var.dims):
		dim = ncfile.createDimension( dname, var.shape[idim] )
		cvar: Variable = ncfile.createVariable( dname, datatype=np.float32, dimensions=(dname,), fill_value=np.nan )
		coord: xa.DataArray = var.coords[dname]
		cvar[:] = coord.values
		cvar.setncatts( coord.attrs )
#		cvar.units = coord.attrs['units']
#		cvar.long_name = coord.attrs['long_name']
	dvar: Variable = ncfile.createVariable(var.name, datatype=var.dtype, dimensions=var.dims, fill_value=np.nan)
	print( f" nc4_write_array: ")
	print(f"  >>  var:  shape = {var.shape} ({var.values.shape})")
	print(f"  >>  dvar: shape = {dvar.shape}, dims = {dvar.get_dims()}")
	if   var.ndim == 1:   dvar[:] = var.values
	elif var.ndim == 2:   dvar[:,:] = var.values
	elif var.ndim == 3:   dvar[:,:,:] = var.values
	elif var.ndim == 4:   dvar[:,:,:,:] = var.values
	dvar.setncatts( var.attrs )
	ncfile.close()