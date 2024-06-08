import torch
import xarray as xa
import hydra, os
from fmod.base.util.config import fmconfig, ConfigContext
from fmod.controller.dual_trainer import ModelTrainer
from fmod.base.io.loader import TSet
from fmod.model.sres.manager import SRModels

hydra.initialize(version_base=None, config_path="../config")
device = ConfigContext.set_device()

task = "sres"
models = ['mscnn' ] # 'dbpn', 'edsr', 'srdn', 'unet', 'vdsr', 'mscnn' ]
dataset = "LLC4320-v1"
scenario = "s4.1"
load_state = "current"
save_state = True

for model in models:
	with ConfigContext(task, model, dataset, scenario) as ccfg:

		model_manager: SRModels = SRModels( device )
		trainer: ModelTrainer = ModelTrainer(model_manager)

		for tset in [ TSet.Train, TSet.Validation, TSet.Test ]:
			trainer.evaluate( tset )






