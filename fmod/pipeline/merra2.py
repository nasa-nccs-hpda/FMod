import numpy as np, xarray as xa
import torch, dataclasses
import nvidia.dali.plugin.pytorch as dali_pth
from dataclasses import dataclass
from datetime import date, timedelta
import nvidia.dali as dali
from fmod.base.util.dates import date_list, year_range
from fmod.base.util.config import cfg2meta, cfg
from typing import Iterable, List, Tuple, Union, Optional, Dict
from modulus.datapipes.datapipe import Datapipe
from fmod.base.source.merra2.model import FMBatch, BatchType
from modulus.datapipes.meta import DatapipeMetaData
from fmod.base.source.merra2 import batch
Tensor = torch.Tensor

@dataclass
class MetaData(DatapipeMetaData):
    name: str = "MERRA2NC"
    # Optimization
    auto_device: bool = True
    cuda_graphs: bool = True
    # Parallel
    ddp_sharding: bool = True

pmeta: MetaData =cfg2meta('pipeline', MetaData(), on_missing="skip" )

class MERRA2InputIterator(object):
    def __init__(self):
        self.train_steps = cfg().task.train_steps
        self.dts = cfg().task.data_timestep
        self.n_day_offsets = 24//self.dts
        self.target_lead_times = [f"{iS * self.dts}h" for iS in range(1, self.train_steps + 1)]
        self.train_dates = year_range(*cfg().task.year_range, randomize=True)
        self.nepochs = cfg().task.nepoch
        self.max_iter = cfg().task.max_iter
        self.fmbatch: FMBatch = FMBatch(BatchType.Training)
        self.norms: Dict[str, xa.Dataset] = self.fmbatch.norm_data
        self.current_date = date(0,0,0 )
        self.mu: xa.Dataset  = self.norms['mean_by_level']
        self.sd: xa.Dataset  = self.norms['stddev_by_level']
        self.dsd: xa.Dataset = self.norms['diffs_stddev_by_level']

    def __iter__(self):
        self.i = 0
        self.length = len(self.train_dates)*self.n_day_offsets
        return self

    def get_date(self):
        return self.train_dates[ self.i // self.n_day_offsets ]

    def get_day_offset(self):
        return self.i % self.n_day_offsets

    def __next__(self):
        next_date = self.get_date()
        if self.current_date != next_date:
            self.fmbatch.load( next_date )
            self.current_date = next_date
        train_data: xa.Dataset = self.fmbatch.get_train_data( self.get_day_offset() )
        (inputs, targets, forcings) = batch.extract_inputs_targets_forcings(train_data, target_lead_times=self.target_lead_times, **dataclasses.asdict(cfg().task))
        self.i = (self.i + 1) % self.length
        return inputs, targets, forcings

class MERRA2NCDatapipe(Datapipe):
    """MERRA2 DALI data pipeline for NetCDF files"""


    def __init__(self):
        super().__init__(meta=pmeta)
        self.num_workers: int = cfg().platform.num_workers
        self.device = self.get_device()
        self.pipe = self._create_pipeline()
        self.batch_size = 1

    @classmethod
    def get_device(cls) -> torch.device:
        device = torch.device( cfg().platform.device )
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda:0")
        return  device

    def _create_pipeline(self) -> dali.Pipeline:
        pipe = dali.Pipeline(
            batch_size=self.batch_size,
            num_threads=2,
            prefetch_queue_depth=2,
            py_num_workers=self.num_workers,
            device_id=self.device.index,
            py_start_method="spawn",
        )

        with pipe:
            source = MERRA2InputIterator()
            self.length = source.length
            invar, outvar, forcing = dali.fn.external_source( source, num_outputs=3, parallel=True, batch=False )
            if self.device.type == "cuda":
                invar = invar.gpu()
                outvar = outvar.gpu()

            invar = dali.fn.normalize( invar, mean=source.mu, stddev=source.sd )
            outvar = dali.fn.normalize( outvar, mean=source.mu, stddev=source.sd )
            pipe.set_outputs(invar, outvar)

        return pipe

    def __iter__(self):
        self.pipe.reset()
        return dali_pth.DALIGenericIterator([self.pipe], [ "invar", "outvar", "forcings"])

    def __len__(self):
        return self.length
