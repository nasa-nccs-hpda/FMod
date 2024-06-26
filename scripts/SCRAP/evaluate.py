import torch
import xarray as xa
import hydra, os
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmod.base.util.config import fmconfig, ConfigContext
from fmod.controller.dual_trainer import ModelTrainer
from fmod.base.io.loader import TSet
from fmod.model.sres.manager import SRModels, ResultsAccumulator

hydra.initialize(version_base=None, config_path="../../config")
device = ConfigContext.set_device()

task = "sres"
models = [ 'dbpn', 'edsr', 'srdn', 'unet', 'vdsr', 'mscnn' ]
dataset = "LLC4320-v1"
scenario = "s4.1"

for model in models:
	with ConfigContext(task, model, dataset, scenario) as cc:
		results = ResultsAccumulator(task, dataset, scenario, model)
		model_manager: SRModels = SRModels( device )
		trainer: ModelTrainer = ModelTrainer(model_manager)

		for tset in [TSet.Test]:
			losses: Dict[str,float] = trainer.evaluate( tset )
			results.record_losses( tset, 0, losses['validation'], losses['upsampled'] )

		results.save( )






