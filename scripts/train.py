import torch
import xarray as xa
import hydra, os
from fmod.base.util.config import fmconfig, Configuration
from fmod.controller.dual_trainer import ModelTrainer
from fmod.base.io.loader import TSet
from fmod.model.sres.manager import SRModels
from fmod.data.batch import BatchDataset
from fmod.base.source.loader import srRes
hydra.initialize(version_base=None, config_path="../config")

task = "sres"
models = [ 'dbpn' ]
dataset = "LLC4320-v1"
scenario = "s4.1"
load_state = "current"
save_state = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
	torch.cuda.set_device(device.index)

for model in models:
	fmconfig(task, model, dataset, scenario)

	model_manager: SRModels = SRModels( device )
	trainer: ModelTrainer = ModelTrainer(model_manager)
	sample_input: xa.DataArray = model_manager.get_sample_input()
	sample_target: xa.DataArray = model_manager.get_sample_target()

	trainer.train(load_state=load_state, save_state=save_state)

	Configuration.clear()




