import numpy as np, xarray as xa
from fmod.base.util.config import ConfigContext, cfg
from fmod.base.source.loader.raw import SRRawDataLoader

cname: str = "sres"
model: str =  'rcan-10-20-64'
task = "swot"
dataset = "swot"
platform = "explore"

ConfigContext.set_defaults( platform=platform, task=task, dataset=dataset )
with ConfigContext(cname, model=model ) as cc:
	loader: SRRawDataLoader = SRRawDataLoader.get_loader( cfg().task )
	ns = loader.norm_stats
	print( {vn: nv.shape for vn,nv in ns.items()} )



