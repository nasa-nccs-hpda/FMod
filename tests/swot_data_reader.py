import numpy as np, xarray as xa
from fmod.base.util.config import ConfigContext, cfg
from fmod.base.source.loader.raw import SRRawDataLoader

cname = "sres"
model = "rcan"
platform = "explore"
task = "swot_1x1"
dataset = "swot"
file_index  =  1425024
varname="SST"

ConfigContext.set_defaults( platform=platform, task=task, dataset=dataset )
with ConfigContext(cname, model=model ) as cc:
	loader: SRRawDataLoader = SRRawDataLoader.get_loader( cfg().task )
	data: xa.DataArray = loader.load_file( varname=varname, index=file_index )
	print( data.shape )



