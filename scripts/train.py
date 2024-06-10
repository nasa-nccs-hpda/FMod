import torch
import xarray as xa
import hydra, os
from fmod.base.util.config import fmconfig, ConfigContext
from fmod.controller.dual_trainer import ModelTrainer
from fmod.base.io.loader import TSet
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmod.model.sres.manager import SRModels, ResultsAccumulator
from fmod.data.batch import BatchDataset
from fmod.base.source.loader import srRes
hydra.initialize(version_base=None, config_path="../config")
device = ConfigContext.set_device()

task = "sres"
models = ['dbpn' ] # 'dbpn', 'edsr', 'srdn', 'unet', 'vdsr', 'mscnn' ]
dataset = "LLC4320-v1"
scenario = "s4.1"

results = ResultsAccumulator(task,dataset,scenario)
for model in models:

	with ConfigContext(task, model, dataset, scenario) as cc:

		model_manager: SRModels = SRModels( device )
		trainer: ModelTrainer = ModelTrainer(model_manager)
		trainer.train()

		for tset in [TSet.Validation, TSet.Test]:
			losses: Dict[str,float] = trainer.evaluate( tset )
			results.record_losses( model, tset, losses['validation'], losses['upsampled'] )

		results.save( cc.cfg.platform.processed )

results.print()






