import torch, time
import xarray as xa
from fmod.base.gpu import set_device, get_device
import hydra, os
from fmod.base.util.config import fmconfig, ConfigContext, cfg
from fmod.controller.dual_trainer import ModelTrainer
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmod.model.sres.manager import SRModels, ResultsAccumulator
from fmod.base.util.ops import fmbdir, fmtp


task = "sres"
models = [ 'dbpn' ] # [ 'dbpn', 'edsr', 'srdn', 'unet', 'vdsr', 'mscnn', 'lapsrn' ]
dataset = "LLC4320-v1"
scenario = "s4"
refresh_state = False
gpu = 0
seed = int( time.time()/60 )
ccustom = { 'task.nepochs': 1 }

for model in models:
	with ConfigContext(model, dataset, scenario, ccustom ) as cc:
		cfg().pipeline['gpu'] = gpu
		t0 = time.time()
		results = ResultsAccumulator(task, dataset, scenario, model, refresh_state=refresh_state)
		model_manager: SRModels = SRModels( set_device() )
		trainer: ModelTrainer = ModelTrainer( model_manager, results )
		trainer.train( refresh_state=refresh_state, seed=seed )
		results.save()
		print( f" ******** Model '{model}' completed {cfg().task.nepochs} epochs of training in {(time.time()-t0)/60:.2f} min ******** ")
		results.rprint()






