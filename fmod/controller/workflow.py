import torch, time
from fmod.base.util.config import ConfigContext, cfg
from fmod.controller.dual_trainer import ModelTrainer
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmod.model.sres.manager import ResultsAccumulator

class WorkflowController(object):

	def __init__(self, cname: str, configuration: Dict[str,Any], **kwargs):
		self.cname = cname
		self.seed = kwargs.get('seed', int( time.time()/60 ) )
		self.refresh_state = kwargs.get('refresh_state', False )
		ConfigContext.set_defaults( **configuration )

	def train(self, models: List[str], **kwargs):
		for model in models:
			with ConfigContext(self.cname, model=model, **kwargs) as cc:
				trainer: ModelTrainer = ModelTrainer(cc)
				trainer.train(refresh_state=self.refresh_state, seed=self.seed)

