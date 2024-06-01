import torch
import xarray as xa, numpy as np
import hydra, os
from datetime import datetime
from typing import Dict, List
from fmod.view.sres import mplplot, create_plot_data
from fmod.base.util.config import fmconfig, cfg
from fmod.controller.dual_trainer import ModelTrainer
from fmod.controller.dual_trainer import LearningContext
from fmod.model.sres.manager import SRModels
from fmod.data.batch import BatchDataset
hydra.initialize(version_base=None, config_path="../config")

task = "sres"
model = "vdsr"
dataset = "LLC4320"
scenario = "s1"
fmconfig(task, model, dataset, scenario)
# lgm().set_level( logging.DEBUG )

load_state = "current"
save_state = True
cfg().task['nepochs'] = 1
eval_tileset = LearningContext.Validation

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
	torch.cuda.set_device(device.index)

input_dataset: BatchDataset = BatchDataset(cfg().task, vres="low", )
target_dataset: BatchDataset = BatchDataset(cfg().task, vres="high")

model_manager: SRModels = SRModels(input_dataset, target_dataset, device)
trainer: ModelTrainer = ModelTrainer(model_manager)
sample_input: xa.DataArray = model_manager.get_sample_input()
sample_target: xa.DataArray = model_manager.get_sample_target()

# Training the model

train_losses: Dict[str, float] = trainer.train(load_state=load_state, save_state=save_state)
print( f"Completed Training, loss = {train_losses['predictions']:.3f}")

# Validating the Model

eval_losses = trainer.evaluate(eval_tileset)
print( f"Completed Validation:")
print( f" * validation loss = {eval_losses['predictions']:.3f}")
print( f" * upsampled  loss = {eval_losses['upsampled']:.3f}")


