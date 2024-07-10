import math, random
from typing import Dict, Tuple, List, Optional
from fmod.base.io.loader import TSet, batchDomain
from fmod.base.util.config import cfg

class TileIterator(object):

    def __init__(self, tset: TSet = TSet.Train, **kwargs ):
        self.grid = TileGrid(tset)
        self.randomize: bool = kwargs.get('randomize', False)
        self.regular_grid: List[  Dict[str,int]  ] = list( self.grid.get_tile_locations(**kwargs).values() )
        self.domain: batchDomain = batchDomain.from_config( cfg().task.get('batch_domain', 'tiles'))
        self.batch_size: int = cfg().task.batch_size[tset.value]
        self.ntiles = kwargs.get('ntiles', 0)
        self.index: int = 0

    def __iter__(self):
        if self.randomize: random.shuffle( self.regular_grid )
        self.index = 0
        return self

    @property
    def active(self):
        if self.domain == batchDomain.Time:
            return self.index < len(self.regular_grid)
        elif self.domain == batchDomain.Tiles:
            return (self.ntiles == 0) or (self.index < self.ntiles)

    def __next__(self) ->  Dict[str,int]:
        if not self.active: raise StopIteration()
        if self.domain == batchDomain.Time:
            result = self.regular_grid[self.index]
            self.index = self.index + 1
            return result
        elif self.domain == batchDomain.Tiles:
            result = dict( start=self.index, end=self.index + self.batch_size )
            self.index = self.index + self.batch_size
            return result

class TileGrid(object):

    def __init__(self, tset: TSet = TSet.Train):
        self.tset: TSet = tset
        origins: Dict[str,Dict[str,int]] = cfg().task.get('origin',{})
        # print( f"TileGrid: origins={list(origins.keys())}, tset='{self.tset.value}'")
        self.origin: Dict[str,int] = origins[self.tset.value]
        self.tile_grid: Dict[str, int] = None
        self.tile_size: Dict[str,int] = cfg().task.tile_size
        self.tlocs: Dict[Tuple[int,int],Dict[str,int]] = {}
        downscale_factors: List[int] = cfg().model.downscale_factors
        self.downscale_factor = math.prod(downscale_factors)

    def get_global_grid_shape(self, image_shape: Dict[str, int]):
        ts = self.get_full_tile_size()
        global_shape = {dim: image_shape[dim] // ts[dim] for dim in ['x', 'y']}
        return global_shape

    def get_grid_shape(self, image_shape: Dict[str, int]) -> Dict[str, int]:
        global_grid_shape = self.get_global_grid_shape(image_shape)
        cfg_grid_shape = cfg().task.tile_grid[self.tset.value]
        self.tile_grid = { dim: (cfg_grid_shape[dim] if (cfg_grid_shape[dim]>=0) else global_grid_shape[dim]) for dim in ['x', 'y'] }
        return self.tile_grid

    def get_active_region(self, image_shape: Dict[str, int] ) -> Dict[str, Tuple[int,int]]:
        ts = self.get_full_tile_size()
        gs = self.get_grid_shape( image_shape )
        print( f"get_active_region: gs={gs}, ts={ts}" )
        region = { d: (self.origin[d],self.origin[d]+ts[d]*gs[d]) for d in ['x', 'y'] }
        return region

    def get_tile_size(self, downscaled: bool = False ) -> Dict[str, int]:
        sf = self.downscale_factor if downscaled else 1
        rv = { d: self.tile_size[d] * sf for d in ['x', 'y'] }
        return  rv

    def get_full_tile_size(self) -> Dict[str, int]:
        return { d: self.tile_size[d] * self.downscale_factor for d in ['x', 'y'] }

    def get_tile_origin( self, ix: int, iy: int, downscaled: bool = False ) -> Dict[str, int]:
        sf = self.downscale_factor if downscaled else 1
        return { d: self.origin[d] + self.cdim(ix, iy, d) * self.tile_size[d] * sf for d in ['x', 'y'] }

    def get_tile_locations(self, **kwargs ) -> Dict[ Tuple[int,int], Dict[str,int] ]:
        downscaled: bool = kwargs.get('downscaled', False)
        selected_tile: Optional[Tuple[int,int]] = kwargs.get('selected_tile', None)
        if len(self.tlocs) == 0:
            if self.tile_grid is None:
                self.tile_grid = cfg().task.tile_grid[self.tset.value]
            for ix in range(self.tile_grid['x']):
                for iy in range(self.tile_grid['y']):
                    if (selected_tile is None) or ((ix,iy) == selected_tile):
                        self.tlocs[(ix,iy)] = self.get_tile_origin(ix,iy,downscaled)
        return self.tlocs



    @classmethod
    def cdim(cls, ix: int, iy: int, dim: str) -> int:
        if dim == 'x': return ix
        if dim == 'y': return iy
