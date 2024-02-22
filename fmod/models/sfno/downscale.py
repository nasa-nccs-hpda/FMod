from fmod.base.source.merra2.model import load_const_dataset, load_merra2_norm_data, load_dataset
from fmod.base.util.ops import print_norms, vars3d
from xarray.core.resample import DataArrayResample
import xarray as xa, pandas as pd
import numpy as np
from fmod.base.util.config import cfg
from xarray.core.types import T_DataArray
from fmod.base.util.model import dataset_to_stacked
from typing import List, Union, Tuple, Optional, Dict, Type, Any, Sequence, Mapping, Literal, Hashable
import glob, sys, os, time, traceback
from datetime import date
from fmod.base.util.ops import get_levels_config, increasing, replace_nans
from fmod.base.util.logging import lgm, exception_handled, log_timing
from fmod.base.source.merra2.model import merge_batch
from fmod.base.io.loader import ncFormat
from enum import Enum
from fmod.pipeline.rescale import DataLoader, QType
from torch_harmonics import *
import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np

class SFNODownscaler(object):

	def __init__(self, n_theta: int, n_lambda: int, **kwargs):
		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.sht = RealSHT(n_theta, n_lambda, grid="equiangular").to(device)
		self.isht = InverseRealSHT(n_theta, n_lambda, grid="equiangular").to(device)


	def process(self, variable: xa.DataArray, target: xa.DataArray, qtype: QType) -> xa.DataArray:
		xc, yc = target.coords[self.cn['x']], target.coords[self.cn['y']]

		return target