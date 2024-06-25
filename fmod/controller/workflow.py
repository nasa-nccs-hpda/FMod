import torch, time
from fmod.base.util.config import ConfigContext, cfg
from fmod.controller.dual_trainer import ModelTrainer
from fmod.base.io.loader import ncFormat, TSet
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmod.base.plot.sres import SRPlot

class WorkflowController(object):

	def __init__(self, cname: str, configuration: Dict[str,Any], **kwargs):
		self.cname = cname
		self.seed = kwargs.get('seed', int( time.time()/60 ) )
		self.refresh_state = kwargs.get('refresh_state', False )
		self.config: ConfigContext = None
		self.trainer: ModelTrainer = None
		self.plot: Any = None
		ConfigContext.set_defaults( **configuration )

	def train(self, models: List[str], **kwargs):
		for model in models:
			with ConfigContext(self.cname, model=model, **kwargs) as cc:
				self.config = cc
				self.trainer = ModelTrainer(cc)
				self.trainer.train(refresh_state=self.refresh_state, seed=self.seed)

	def plot_init(self, cname, **kwargs ):
		self.config = ConfigContext.activate_global(cname, **kwargs )
		self.trainer = ModelTrainer( self.config )

	def get_result_view(self, tset: TSet, **kwargs ):
		self.plot = SRPlot(self.trainer, tset, **kwargs )
		return self.plot.plot()

