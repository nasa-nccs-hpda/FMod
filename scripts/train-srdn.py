import torch, time
import xarray as xa
from fmod.base.gpu import set_device, get_device
import hydra, os
from fmod.base.util.config import ConfigContext, cfg
from fmod.controller.dual_trainer import ModelTrainer
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmod.model.sres.manager import SRModels, ResultsAccumulator


refresh_state = False
seed = int( time.time()/60 )
cname = "sres"
models = [ 'srdn' ] # [ 'dbpn', 'edsr', 'srdn', 'unet', 'vdsr', 'mscnn', 'lapsrn' ]

config = dict(
	task = "sres",
	dataset = "LLC4320",
	scenario = "s4",
	pipeline = "sres",
	server = "explore"
)
ccustom = { 'task.nepochs': 30, 'pipeline.gpu': 0 }

for model in models:
	config['model'] = model
	with ConfigContext( cname, config, ccustom ) as cc:
		t0 = time.time()
		results = ResultsAccumulator( refresh_state=refresh_state, **config )
		model_manager: SRModels = SRModels( set_device() )
		trainer: ModelTrainer = ModelTrainer( model_manager, results )
		trainer.train( refresh_state=refresh_state, seed=seed )
		results.save()
		print( f" ******** Model '{model}' completed {cfg().task.nepochs} epochs of training in {(time.time()-t0)/60:.2f} min ******** ")
		results.rprint()






