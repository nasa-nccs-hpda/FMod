# Documentation of the variables: https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf

from pydap.client import open_url
import xarray as xr
import numpy as np
import datetime


def get_fourcastnet_grids():
	fourcastnet_lat = np.linspace(-90, 90, 720)
	fourcastnet_lon = np.linspace(-180, 180, 1440)
	return fourcastnet_lat, fourcastnet_lon

def get_dataset(opendap_url, sessions, variables=None):
    opendap_data = open_url(opendap_url, session=sessions)
    dataset = xr.open_dataset(xr.backends.PydapDataStore(opendap_data[variables]))
    return dataset


def update_levels(list_of_vars):
    list_of_vars_updated = []
    for variables, levels in zip(list_of_vars, np.arange(0, len(list_of_vars), 1)):
        list_of_vars_updated.append(variables.assign_coords({"lev": levels}))
    return list_of_vars_updated


def get_merra_urls(timestamp):
    dtime = datetime.strptime(timestamp, "%Y%m%d")
    url_prefix1 = "https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/"
    url_prefix2 = "https://goldsmr5.gesdisc.eosdis.nasa.gov/opendap/MERRA2/"

    surface_url = f"{url_prefix1}M2I1NXASM.5.12.4/{dtime.strftime('%Y/%m/')}MERRA2_401.inst1_2d_asm_Nx.{dtime.strftime('%Y%m%d')}.nc4"
    UV_url = f"{url_prefix2}M2I3NPASM.5.12.4/{dtime.strftime('%Y/%m/')}MERRA2_401.inst3_3d_asm_Np.{dtime.strftime('%Y%m%d')}.nc4"
    H_url = f"{url_prefix2}M2I6NPANA.5.12.4/{dtime.strftime('%Y/%m/')}MERRA2_401.inst6_3d_ana_Np.{dtime.strftime('%Y%m%d')}.nc4"
    TCWV_url = f"{url_prefix1}M2T1NXINT.5.12.4/{dtime.strftime('%Y/%m/')}MERRA2_401.tavg1_2d_int_Nx.{dtime.strftime('%Y%m%d')}.nc4"
    return surface_url, UV_url, H_url, TCWV_url


def extract_vars_from_url(session, surface_url, UV_url, H_url, TCWV_url):
    print("Extracting Surface variables ...")
    sfc_dataset = get_dataset(
        surface_url,
        session,
        variables=("U10M", "V10M", "T2M", "PS", "SLP", "lat", "lon", "lev", "time"),
    ).isel(time=np.arange(0, 24, 6))

    print("Extracting U, V, T and RH ...")
    UVTRH_dataset = (
        get_dataset(
            UV_url,
            session,
            variables=("U", "V", "T", "RH", "lat", "lon", "lev", "time"),
        )
        .isel(time=np.arange(0, 8, 2))
        .sel(lev=[1000, 850, 500])
    )

    print("Extracting H ...")
    H_dataset = get_dataset(
        H_url, session, variables=("H", "lat", "lon", "lev", "time")
    ).sel(lev=[1000, 850, 500, 50])

    print("Extracting TCWV ...")
    TCWV_dataset = get_dataset(
        TCWV_url,
        session,
        variables=(
            "DQVDT_ANA",
            "DQVDT_CHM",
            "DQVDT_DYN",
            "DQVDT_MST",
            "DQVDT_PHY",
            "DQVDT_TRB",
            "lat",
            "lon",
            "lev",
            "time",
        ),
    ).isel(time=np.arange(0, 24, 6))

    return sfc_dataset, UVTRH_dataset, H_dataset, TCWV_dataset


def interp_variables(
    sfc_dataset,
    UVTRH_dataset,
    H_dataset,
    TCWV_dataset,
):
    fourcastnet_lat, fourcastnet_lon = get_fourcastnet_grids()

    print("Interpolating variables ...")
    # Surface | U10, V10, T2m, sp, mslp
    u10m = (
        sfc_dataset["U10M"]
        .interp(lon=fourcastnet_lon, lat=fourcastnet_lat)
        .assign_coords({"lev": 1})
    )
    v10m = (
        sfc_dataset["V10M"]
        .interp(lon=fourcastnet_lon, lat=fourcastnet_lat)
        .assign_coords({"lev": 2})
    )
    t2m = (
        sfc_dataset["T2M"]
        .interp(lon=fourcastnet_lon, lat=fourcastnet_lat)
        .assign_coords({"lev": 3})
    )
    sp = (
        sfc_dataset["PS"]
        .interp(lon=fourcastnet_lon, lat=fourcastnet_lat)
        .assign_coords({"lev": 4})
    )
    slp = (
        sfc_dataset["SLP"]
        .interp(lon=fourcastnet_lon, lat=fourcastnet_lat)
        .assign_coords({"lev": 5})
    )

    ### MSLP

    # 1000 hPa, 850, 500 | U, V
    U = (
        UVTRH_dataset["U"]
        .sel(lev=[1000, 850, 500])
        .interp(lon=fourcastnet_lon, lat=fourcastnet_lat)
    )
    V = (
        UVTRH_dataset["V"]
        .sel(lev=[1000, 850, 500])
        .interp(lon=fourcastnet_lon, lat=fourcastnet_lat)
    )

    #  850 hPa, 500 hPa | T, RH
    T = (
        UVTRH_dataset["T"]
        .sel(lev=[850, 500])
        .interp(lon=fourcastnet_lon, lat=fourcastnet_lat)
    )
    RH = (
        UVTRH_dataset["RH"]
        .sel(lev=[850, 500])
        .interp(lon=fourcastnet_lon, lat=fourcastnet_lat)
    )

    # 1000 hPa, 850 hPa, 500 hPa,  50 hPa | Z
    H = (
        H_dataset["H"]
        .sel(lev=[1000, 850, 500, 50])
        .interp(lon=fourcastnet_lon, lat=fourcastnet_lat)
    )

    # Integrated | TCWV (Total Column Water Vapor)
    TCWV_merge = (
        TCWV_dataset["DQVDT_ANA"]
        + TCWV_dataset["DQVDT_CHM"]
        + TCWV_dataset["DQVDT_DYN"]
        + TCWV_dataset["DQVDT_MST"]
        + TCWV_dataset["DQVDT_PHY"]
        + TCWV_dataset["DQVDT_TRB"]
    )
    TCWV_dataset = TCWV_dataset.assign({"ITCWV": TCWV_merge})
    ITCWV = (
        TCWV_dataset["ITCWV"]
        .interp(lon=fourcastnet_lon, lat=fourcastnet_lat)
        .assign_coords({"lev": 5})
    )

    ITCWV = ITCWV.assign_coords(time=ITCWV["time"].values - np.timedelta64(30, "m"))

    # Updating levels in all the variables
    # Writing variables in sequence

    variables = (
        u10m,
        v10m,
        t2m,
        sp,
        slp,
        U.sel(lev=1000),
        V.sel(lev=1000),
        H.sel(lev=1000),
        T.sel(lev=850),
        U.sel(lev=850),
        V.sel(lev=850),
        H.sel(lev=850),
        RH.sel(lev=850),
        T.sel(lev=500),
        U.sel(lev=500),
        V.sel(lev=500),
        H.sel(lev=500),
        RH.sel(lev=500),
        H.sel(lev=50),
        ITCWV,
    )
    return variables


def var_to_h5(variables, output_filename="dummy.h5"):
    fourcastnet_variables = update_levels(variables)
    fourcastnet_input = xr.concat((fourcastnet_variables), dim="lev")

    # Deleting attributes
    attributes = list(fourcastnet_input.attrs.keys())
    for attrs in attributes:
        del fourcastnet_input.attrs[attrs]

    fourcastnet_input = fourcastnet_input.to_dataset()
    fourcastnet_input = fourcastnet_input.rename({"U10M": "fields"})
    fourcastnet_input.to_netcdf(output_filename)
