from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmod.controller.workflow import WorkflowController

cname: str = "sres"
models: List[str] = [ 'srdn' ] # [ 'dbpn', 'edsr', 'srdn', 'unet', 'vdsr', 'mscnn', 'lapsrn' ]
ccustom: Dict[str,Any] = { 'task.nepochs': 10, 'task.lr': 2e-4, 'pipeline.gpu': 1 }

configuration = dict(
	task = "cape_basin",
	dataset = "LLC4320",
	pipeline = "sres",
	platform = "explore"
)

controller = WorkflowController( cname, configuration )
controller.train( models, **ccustom )







