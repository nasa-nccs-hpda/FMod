#%%matplotlib ipympl
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmod.controller.workflow import WorkflowController
from fmod.base.io.loader import ncFormat, TSet

cname = "sres"
model =  'dbpn'  # [ 'dbpn', 'edsr', 'srdn', 'unet', 'vdsr', 'mscnn' ]
ccustom: Dict[str,Any] = {}

configuration = dict(
	task = "cape_basin",
	dataset = "LLC4320",
	platform = "explore_gt"
)

import torch
print(torch.cuda.is_available())
controller = WorkflowController( cname, configuration )
controller.init_plotting( cname, model, **ccustom )






