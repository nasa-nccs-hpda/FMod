from fmod.controller.dual_trainer import ModelTrainer
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple, Type, Optional, Callable

class Plot:

	def __init__(self, trainer: ModelTrainer,  **kwargs):
		self.trainer: ModelTrainer = trainer
		self.fsize = kwargs.get('fsize', 8.0)
		self.yscale = kwargs.get('yscale', 'log' )
		self.fig = None
		self.axs = None
		self.aspect = 1

	def create_figure(self, title: str, **kwargs):
		sharex = kwargs.get('sharex', False)
		sharey = kwargs.get('sharey', False)
		nrows  = kwargs.get('nrows', 1)
		ncols  = kwargs.get('ncols', 1)
		callbacks: Dict[str,Callable] = kwargs.get( 'callbacks', {} )
		with plt.ioff():
			self.fig, self.axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[self.fsize*self.aspect, self.fsize], sharex=sharex, sharey=sharey, layout="tight")
			self.fig.suptitle( title, fontsize=14, va="top", y=1.0)
			for event, callback in callbacks.items():
				self.fig.canvas.mpl_connect(event,callback)