import torch
import xarray as xa
import hydra, os
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmod.base.util.config import fmconfig, ConfigContext
from fmod.controller.dual_trainer import ModelTrainer
from fmod.base.io.loader import TSet
from fmod.model.sres.manager import SRModels, ResultsAccumulator

hydra.initialize(version_base=None, config_path="../config")
device = ConfigContext.set_device()

task = "sres"
dataset = "LLC4320-v1"
scenario = "s4.1"
model = "mscnn"
downscale_factors= [ 4 ]
ups_mode= 'bicubic'

results = ResultsAccumulator(task,dataset,scenario)
with ConfigContext(task, model, dataset, scenario) as cc:
	cc.cfg.model.downscale_factors= downscale_factors
	cc.cfg.model.ups_mode = ups_mode

	model_manager: SRModels = SRModels( device )
	trainer: ModelTrainer = ModelTrainer(model_manager)

	for tset in TSet:
		loss: float = trainer.eval_upscale( tset )
		results.record_losses( model, tset, 0.0, loss )

	results.save( cc.cfg.platform.processed )
	results.print()






