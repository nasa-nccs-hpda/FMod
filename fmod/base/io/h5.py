import h5netcdf
import numpy as np
import xarray as xa

def write_array( path: str, var: xa.DataArray ):
    with h5netcdf.File( path, 'w') as f:
        f.dimensions = { var.dims[i]: var.shape[i] for i in range(var.ndim) }
        v = f.create_variable( var.name, dimensions=var.dims, data=var.values, fillvalue=np.nan )
        v.attrs.update(var.attrs)
