import torch
import xarray as xa
import hydra, os
from fmod.base.util.config import fmconfig, ConfigContext
from fmod.controller.dual_trainer import ModelTrainer
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmod.model.sres.manager import SRModels, ResultsAccumulator
from fmod.base.util.ops import fmbdir, fmtp
hydra.initialize(version_base=None, config_path="../config")
device = ConfigContext.set_device()

task = "sres"
models = [ 'mscnn' ]
dataset = "LLC4320-v1"
scenario = "s4"
refresh_state = False
seed = 48332

results = ResultsAccumulator(task,dataset,scenario)
for model in models:
	with ConfigContext(task, model, dataset, scenario) as cc:
		model_manager: SRModels = SRModels( device )
		trainer: ModelTrainer = ModelTrainer( model_manager, results )
		trainer.train( refresh_state=refresh_state, seed=seed )
		results.save( fmbdir('processed') )

results.print()






