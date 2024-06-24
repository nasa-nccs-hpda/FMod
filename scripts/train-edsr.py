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
models = [ 'edsr' ] # [ 'dbpn', 'edsr', 'srdn', 'unet', 'vdsr', 'mscnn', 'lapsrn' ]

ConfigContext.set_defaults(
	task = "cape_basin",
	dataset = "LLC4320",
	pipeline = "sres",
	platform = "explore"
)
ccustom = { 'task.nepochs': 30, 'task.lr': 3e-4, 'pipeline.gpu': 1 }

for model in models:
	with ConfigContext( cname, model=model, **ccustom ) as cc:
		t0 = time.time()
		results = ResultsAccumulator(cc)
		model_manager: SRModels = SRModels( set_device() )
		trainer: ModelTrainer = ModelTrainer( model_manager, results )
		trainer.train( refresh_state=refresh_state, seed=seed )
		results.save()
		print( f" ******** Model '{model}' completed {cfg().task.nepochs} epochs of training in {(time.time()-t0)/60:.2f} min ******** ")
		results.rprint()






