import numpy as np, xarray as xa
from fmod.base.util.config import ConfigContext, cfg
from fmod.base.source.loader.raw import SRRawDataLoader
from fmod.base.source.swot.raw import SWOTRawDataLoader

cname = "sres"
model = "rcan"
platform = "explore"
task = "swot_1x1"
dataset = "swot"

ConfigContext.set_defaults( platform=platform, task=task, dataset=dataset )
with ConfigContext(cname, model=model ) as cc:
	loader: SRRawDataLoader = SRRawDataLoader.get_loader( cfg().task )
	tidxs = loader.get_batch_time_indices()
	print(tidxs)




