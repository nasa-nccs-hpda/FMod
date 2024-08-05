from fmod.controller.dual_trainer import ModelTrainer
import matplotlib.pyplot as plt
import ipywidgets as ipw
from typing import Any, Dict, List, Tuple, Type, Optional, Callable
from abc import ABC, abstractmethod

class Plot(ABC):

	def __init__(self, trainer: ModelTrainer,  **kwargs):
		self.trainer: ModelTrainer = trainer
		self.model = self.trainer.model_name
		self.fsize = kwargs.get('fsize', 8.0)
		self.yscale = kwargs.get('yscale', 'log' )
		self.fig = None
		self.axs = None
		self.aspect = 1

	def create_figure(self, **kwargs):
		sharex = kwargs.get('sharex', False)
		sharey = kwargs.get('sharey', False)
		nrows  = kwargs.get('nrows', 2)
		ncols  = kwargs.get('ncols', 2)
		title  = kwargs.get('title', "")
		callbacks: Dict[str,Callable] = kwargs.get( 'callbacks', {} )
		with plt.ioff():
			self.fig, self.axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[self.fsize*self.aspect, self.fsize], sharex=sharex, sharey=sharey, layout="tight")
			self.fig.suptitle( title, fontsize=14, va="top", y=1.0)
			for event, callback in callbacks.items():
				self.fig.canvas.mpl_connect(event,callback)

	@abstractmethod
	def plot(self)  -> ipw.Box:
		pass