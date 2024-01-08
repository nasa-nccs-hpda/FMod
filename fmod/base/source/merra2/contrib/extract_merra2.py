# Documentation of the variables: https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf

from pydap.cas.urs import setup_session
import numpy as np
from datetime import datetime
import os
from .merra2 import (
    get_merra_urls,
    extract_vars_from_url,
    interp_variables,
    var_to_h5,
)


username = os.environ["EDUSER"]
password = os.environ["EDPSWD"]

timestamp = "20210829"  # YYYYMMDDHH


surface_url, UV_url, H_url, TCWV_url = get_merra_urls(timestamp)
session = setup_session(username, password, check_url=surface_url)

variables = interp_variables(
    extract_vars_from_url(session, surface_url, UV_url, H_url, TCWV_url)
)
var_to_h5(variables, output_filename=f"MERRA_{timestamp}.h5")
