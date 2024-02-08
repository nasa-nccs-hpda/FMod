from typing  import List, Tuple, Union, Optional, Dict, Callable
import ipywidgets as ipw
from fmod.base.util.logging import lgm, exception_handled, log_timing
from traitlets import CInt, link
from ipywidgets import GridspecLayout

class Counter(ipw.DOMWidget):
    value = CInt(0, sync=True)

class StepSlider:

	def __init__(self, label: str, maxval: int, callback: Callable[[int], int], **kwargs):
		self.grid = GridspecLayout(1, 12 )
		self.layout = ipw.Layout(height='auto', width='auto')
		self.value = kwargs.get('ival',0)
		self.maxval = maxval
		self.executable: Callable[[int], int] = callback
		self.counter = Counter()
		self.slider: ipw.IntSlider = ipw.IntSlider(value=self.value, min=0, max=maxval, description=label, layout=self.layout )
		self.slider.observe(self.update, names='value')
		self.button_cback    = ipw.Button(description='<', button_style='info', on_click=self.bplus, layout=self.layout )
		self.button_cforward = ipw.Button(description='>', button_style='info', on_click=self.bminus, layout=self.layout )
		self.box_layout = ipw.Layout(display='flex', align_items='stretch', border='solid', width='100%')

	def bplus(self,name):
		self.counter.value += 1 if self.counter.value < self.maxval else 0
		lgm().log(f"button_plus: {self.counter.value}")

	def bminus(self,name):
		self.counter.value -= 1 if self.counter.value > 0 else 0
		lgm().log(f"button_minus: {self.counter.value}")

	@exception_handled
	def update(self, change):
		self.value = change['new']
		self.executable(self.value)

	def gui(self):
		self.grid[1, :10] = self.slider
		self.grid[1, 10] = self.button_cback
		self.grid[1, 11] = self.button_cforward
		return self.grid
