import math, numpy as np
import xarray as xa
from typing  import List, Tuple, Union, Optional, Dict, Callable
from fmod.base.util.ops import xaformat_timedeltas, print_data_column
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import ipywidgets as ipw
from torch import Tensor
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from fmod.base.util.grid import GridOps
from fmod.base.io.loader import BaseDataset
from fmod.base.util.logging import lgm, exception_handled, log_timing



class StepSlider:

	def __init__(self, label: str, maxval: int, callback: Callable[[int], int], **kwargs):
		self.value = kwargs.get('ival',0)
		self.maxval = maxval
		self.executable: Callable[[int], int] = callback
		self.slider: ipw.IntSlider = ipw.IntSlider(value=self.value, min=0, max=maxval, description=label, layout=ipw.Layout(flex_grow=10) )
		self.slider.observe(self.update, names='value')
		self.button_cback    = ipw.Button(description='<', button_style='info', layout=ipw.Layout(flex_shrink=1), on_click=self.sback )
		self.button_cforward = ipw.Button(description='>', button_style='info', layout=ipw.Layout(flex_shrink=1), on_click=self.sforward )
		self.box_layout = ipw.Layout(display='flex', align_items='stretch', border='solid', width='100%')

	@exception_handled
	def sback(self, b):
		self.value = (self.value - 1) % ( self.maxval + 1 )
		lgm().log(f"sback: {self.value}")
		self.slider.value = self.value

	@exception_handled
	def sforward(self, b):
		self.value = (self.value + 1) % ( self.maxval + 1 )
		lgm().log(f"sforward: {self.value}")
		self.slider.value = self.value

	@exception_handled
	def update(self, change):
		self.value = change['new']
		self.executable(self.value)

	def gui(self):
		return ipw.HBox( [self.slider, self.button_cforward, self.button_cback], layout=self.box_layout )
