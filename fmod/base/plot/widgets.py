from typing  import List, Tuple, Union, Optional, Dict, Callable
import ipywidgets as ipw
from fmod.base.util.logging import lgm, exception_handled, log_timing
from traitlets import CInt, link
from ipywidgets import GridspecLayout

class Counter(ipw.DOMWidget):
    value = CInt(0, sync=True)

class StepSlider:

	def __init__(self, label: str, maxval: int, callback: Callable[[int], int], **kwargs):
		self.value = kwargs.get('ival',0)
		self.bsize = kwargs.get('bsize','30px')
		self.ssize = kwargs.get('ssize', '920px')
		self.maxval = maxval
		self.executable: Callable[[int], int] = callback
		self.counter = Counter()
		self.slider: ipw.IntSlider = ipw.IntSlider(value=self.value, min=0, max=maxval, description=label, layout=ipw.Layout(width=self.ssize, height=self.bsize) )
		self.slider.observe(self.update, names='value')
		self.button_cback    = ipw.Button(description='<', button_style='info', layout=ipw.Layout(width=self.bsize, height=self.bsize) )
		self.button_cforward = ipw.Button(description='>', button_style='info', layout=ipw.Layout(width=self.bsize, height=self.bsize) )
		self.button_cback.on_click(self.bminus)
		self.button_cforward.on_click(self.bplus)

	@exception_handled
	def bplus(self,b):
		lgm().log(f"button_plus clicked!")
		self.counter.value = ( self.counter.value + 1 ) % (self.maxval+1)
		lgm().log(f"button_plus: {self.counter.value}")

	@exception_handled
	def bminus(self,b):
		lgm().log(f"button_minus clicked!")
		self.counter.value  ( self.counter.value - 1 ) % (self.maxval+1)
		lgm().log(f"button_minus: {self.counter.value}")

	@exception_handled
	def update(self, change):
		self.value = change['new']
		self.executable(self.value)

	def gui(self):
		return ipw.HBox([self.slider, self.button_cback, self.button_cforward])
