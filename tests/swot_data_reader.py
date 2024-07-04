import numpy as np
from fmod.base.util.config import ConfigContext, cfg
from fmod.base.source.loader import srRes
from fmod.base.io.loader import ncFormat, TSet
from fmod.base.source.loader import SRRawDataLoader
from fmod.controller.workflow import WorkflowController

cname = "sres"
model = "rcan"
file_index = 1425024

configuration = dict(
	platform = "explore",
	task = "swot_1x1",
	dataset = "swot"
)

with ConfigContext(cname, model=model ) as cc:
	loader: SRRawDataLoader = SRRawDataLoader.get_loader( cfg().task )
	data: np.ndarray = loader.load_file( varname="SST", index=file_index )
	print( data.shape )



