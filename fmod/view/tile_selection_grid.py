import matplotlib.pyplot as plt
import torch, math
import xarray, traceback, random
from datetime import datetime
from torch import Tensor
from typing import Any, Dict, List, Tuple, Union, Sequence
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

class TileSelectionGrid(object):

	def __init__(self, lcontext: LearningContext):
		self.tile_grid: TileGrid = TileGrid(lcontext)
		self.tiles: List[Rectangle] = None

	def create_tile_recs(self, **kwargs):
		refresh = kwargs.get('refresh', False)
		if (self.tiles is None) or refresh:
			self.tiles = []
			tile_locs: List[Dict[str, int]] = self.tile_grid.get_tile_locations()
			[w,h] =  [ self.tile_grid.tile_size[c] for c in ['x','y'] ]
			for tloc in tile_locs:
				xy = (tloc['x'], tloc['y'])
				r = Rectangle( xy, w, h, fill=False, lw=kwargs.get('lw',1), ec=kwargs.get('color','b') )
				r.set_picker(True)
				self.tiles.append( r )

	def onpick(self,event):
		rect: Rectangle = event.artist
		print( f'Selected rect: ({rect.get_x()}, {rect.get_y()})')

	def overlay_grid(self, ax: plt.Axes, **kwargs):
		self.create_tile_recs(**kwargs)
		p = PatchCollection( self.tiles, alpha=kwargs.get('aplha',0.4) )
		ax.add_collection(p)
		ax.figure.canvas.mpl_connect('pick_event', self.onpick )