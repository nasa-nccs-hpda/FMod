import torch
import hydra, os, time
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping
from fmod.base.util.dates import date_list
from fmod.base.util.config import configure, cfg, cfg_date, cfg2args, pp
from fmod.models.corrdiff.dataset import M2DownscalingDataset

hydra.initialize(version_base=None, config_path="../config")
configure('merra2-sr')

# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(device.index)

train_dates = date_list(cfg_date('task'), cfg().task.max_days)

dataset = M2DownscalingDataset(train_dates=train_dates)

targets, inputs, label = dataset[0]

for i, inp in enumerate(inputs):
    print( f" input-{i}:  {inp.dims} {inp.shape}")

for i, t in enumerate(targets):
    print( f" target-{i}:  {t.dims} {t.shape}")