import numpy as np

def rearrange(d,nx):
	deast = np.c_[d[:nx * nx * 3].reshape(3 * nx, nx),
	d[nx * nx * 3:nx * nx * 6].reshape(3 * nx, nx)]
	dwest = d[nx * nx * 7:].reshape(nx * 2, nx * 3)
	return deast, dwest

def mds2d(dd, nx=4320):
    """
    Reshape an LLC grid data array into separate east and west hemisphere arrays.

    This function takes an array representing data on the LLC grid and separates it into two arrays,
    one for the eastern hemisphere (tiles 1-6) and one for the western hemisphere (tiles 8-13). Tile 7 is for Arctic but not included here.
    The function does not perform any rotation on the data, meaning that the u and v components
    (typically representing eastward and northward velocity components in oceanographic data) remain mixed for the llc grid layout

    Parameters:
    -----------
    dd : numpy.ndarray or list of numpy.ndarray
        The input data array(s) to be reshaped. Each array should be of size 13*nx**2,
        where nx is the size of the LLC grid. The input can also be a list of such arrays.
    nx : int, optional
        The size of one side of the LLC grid. Default is 4320, suitable for the llc4320 model.

    Returns:
    --------
    (deast, dwest) : tuple
        deast : numpy.ndarray or list of numpy.ndarray
            The eastern hemisphere data array(s), each with dimensions (4320x3, 4320x2).
        dwest : numpy.ndarray or list of numpy.ndarray
            The western hemisphere data array(s), each with dimensions (4320x2, 4320x3),
            non-rotated (x is latitude, y is longitude).

    Examples:
    ---------
    # Reshaping a single data array
    east, west = mds2d(data_array)

    # Reshaping a list of data arrays
    reshaped_data_list = mds2d(list_of_data_arrays)

    Notes:
    ------
    The input data should be structured with specific tiles corresponding to different parts of
    the global grid. The function assumes that tiles 1-6 represent the eastern hemisphere and
    tiles 8-13 represent the western hemisphere. Tiles 7 Arctic is not included.
    """

    if type(dd)==type([1]):
        dout=[]
        for d in dd:
            dout.append(rearrange(d,nx))
        return dout
    else:
        return rearrange(dd,nx)