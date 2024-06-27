from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmod.controller.workflow import WorkflowController
import hydra

cname: str = "sres"
models: List[str] = [ 'esrt' ]
ccustom: Dict[str,Any] = { 'task.nepochs': 30, 'task.lr': 1e-4, 'pipeline.gpu': 0 }

configuration = dict(
	task = "cape_basin_1x1",
	dataset = "LLC4320",
	pipeline = "sres",
	platform = "explore"
)

controller = WorkflowController( cname, configuration )
controller.train( models, **ccustom )







