import matplotlib.pyplot as plt
import torch, math
import xarray, traceback, random
from datetime import datetime
from torch import Tensor
from typing import Any, Dict, List, Tuple, Union, Sequence, Callable
from fmod.base.util.config import cdelta, cfg, cval, get_data_coords
from fmod.base.util.array import array2tensor
from fmod.controller.dual_trainer import TileGrid
from fmod.base.util.logging import lgm, exception_handled
from fmod.base.util.ops import pctnan, pctnant
from omegaconf import DictConfig, OmegaConf
from enum import Enum
import numpy as np, xarray as xa
import torch.nn as nn
import time
from fmod.controller.dual_trainer import LearningContext
from matplotlib.collections import PatchCollection
from matplotlib.patches import  Rectangle, Patch

def default_selection_callabck( tilerec: Dict[str,float]):
	print( f" **** Tile selection: {tilerec} ****")

def r2str( r: Rectangle ) -> str:
	return f"({r.get_x()},{r.get_y()})x({r.get_width()},{r.get_height()})"

class TileSelectionGrid(object):

	def __init__(self, lcontext: LearningContext):
		self.tile_grid: TileGrid = TileGrid(lcontext)
		self.tiles: List[Rectangle] = None
		self._selection_callback = default_selection_callabck

	def create_tile_recs(self, **kwargs):
		refresh = kwargs.get('refresh', False)
		randomized = kwargs.get('randomized', False)
		downscaled = kwargs.get('downscaled', True)
		ts: Dict[str, int] = self.tile_grid.get_tile_size(downscaled)
		if (self.tiles is None) or refresh:
			self.tiles = []
			tile_locs: List[Dict[str, int]] = self.tile_grid.get_tile_locations(randomized,downscaled)
			for tloc in tile_locs:
				xy = (tloc['x'], tloc['y'])
				r = Rectangle( xy, ts['x'], ts['y'], fill=False, linewidth=kwargs.get('lw',2), edgecolor=kwargs.get('color','b'), alpha=0.0 )
				r.set_picker(True)
				self.tiles.append( r )

	def set_selection_callabck(self, selection_callabck: Callable):
		self._selection_callback = selection_callabck

	def onpick(self,event):
		print( f" **** Tile selection: {event} ****")
		rect: Rectangle = event.artist
		coords = dict(x=rect.get_x(), y=rect.get_y())
		print(f" ----> Coords: {coords}")
		self._selection_callback( coords )

	def overlay_grid(self, ax: plt.Axes, **kwargs):
		self.create_tile_recs(**kwargs)
		print( f" %%%%  TileSelectionGrid:  overlay grid with {len(self.tiles)} tiles %%%% ")
		print( f" ---> Tiles: {[r2str(t) for t in self.tiles]}")
		p = PatchCollection( self.tiles, alpha=kwargs.get('aplha',0.4) )
		ax.add_collection(p)
		ax.figure.canvas.mpl_connect('pick_event', self.onpick )