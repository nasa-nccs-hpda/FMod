import logging, torch
from fmod.base.util.logging import lgm, exception_handled, log_timing
import xarray as xa
import hydra, os, time
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping
from fmod.base.util.dates import date_list
from fmod.base.util.config import configure, cfg, cfg_date, cfg2args, pp
from fmod.models.sres.srdn.network import SRDN
from fmod.plot.training_results import ResultsPlotter
from fmod.pipeline.trainer import DualModelTrainer
from fmod.pipeline.merra2 import MERRA2Dataset

hydra.initialize(version_base=None, config_path="../config")
configure('merra2-srdn-s1')
# lgm().set_level( logging.DEBUG )

load_state = False
save_state = True
input_res = "low"
target_res = "high"
etype = "l2" # "spectral-l2"

# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(device.index)

input_dataset  = MERRA2Dataset( train_dates=date_list( cfg_date('task'), cfg().task.max_days ), vres=input_res, load_inputs=True, load_base=True )
target_dataset = MERRA2Dataset( train_dates=date_list( cfg_date('task'), cfg().task.max_days ), vres=target_res, load_targets=True )
trainer = DualModelTrainer( input_dataset, target_dataset, device )
sample_input, sample_target, sample_base = next(iter(trainer))
print( f"sample_input{sample_input.dims}: {sample_input.shape} {list(sample_input.coords.keys())}" )
for k,c in sample_input.coords.items():
    print( f"{k}{c.shape}: {c[0]:.2f} -> {c[-1]:.2f}")

inchannels: int = sample_input.shape[0]
nfeatures: Dict[str, int] = cfg().model.nfeatures
nrlayers: int = cfg().model.nrlayers
scale_factors: List[int] = cfg().model.scale_factors
usmethod: str = cfg().model.usmethod
kernel_size: Dict[str, int] = cfg().model.kernel_size

model = SRDN(inchannels, nfeatures, nrlayers, scale_factors, usmethod, kernel_size).to(device)

trainer.train( model, load_state=load_state, save_state=save_state )
inputs, targets, predictions, interpolates = trainer.inference( etype=etype )