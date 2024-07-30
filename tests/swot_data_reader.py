import numpy as np, xarray as xa
from fmod.base.util.config import ConfigContext, cfg
from fmod.base.source.loader.raw import SRRawDataLoader

cname: str = "sres"
model: str =  'rcan-10-20-64'
task = "swot"
dataset = "swot_southpacific_1200"
platform = "explore"

ConfigContext.set_defaults( platform=platform, task=task, dataset=dataset )
with ConfigContext(cname, model=model ) as cc:
	loader: SRRawDataLoader = SRRawDataLoader.get_loader( cfg().task )
	ns: xa.Dataset = loader.global_norm_stats
	for vn, vdata in ns.data_vars.items():
		print( f" {vn}: {vdata.values.tolist()}")



