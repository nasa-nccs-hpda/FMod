import numpy as np, xarray as xa
import torch, dali, dataclasses
import nvidia.dali.plugin.pytorch as dali_pth
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
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

class MERRA2NCDatapipe(Datapipe):
    """MERRA2 DALI data pipeline for NetCDF files

    Parameters
    ----------
    data_dir : str
        Directory where MERRA2 data is stored
    stats_dir : Union[str, None], optional
        Directory to data statistic numpy files for normalization, if None, no normalization
        will be used, by default None
    channels : Union[List[int], None], optional
        Defines which MERRA2 variables to load, if None will use all in NC file, by default None
    batch_size : int, optional
        Batch size, by default 1
    stride : int, optional
        Number of steps between input and output variables. For example, if the dataset
        contains data at every 6 hours, a stride 1 = 6 hour delta t and
        stride 2 = 12 hours delta t, by default 1
    num_steps : int, optional
        Number of timesteps are included in the output variables, by default 1
    patch_size : Union[Tuple[int, int], int, None], optional
        If specified, crops input and output variables so image dimensions are
        divisible by patch_size, by default None
    num_samples_per_year : int, optional
        Number of samples randomly taken from each year. If None, all will be use, by default None
    shuffle : bool, optional
        Shuffle dataset, by default True
    num_workers : int, optional
        Number of workers, by default 1
    device: Union[str, torch.device], optional
        Device for DALI pipeline to run on, by default cuda
    process_rank : int, optional
        Rank ID of local process, by default 0
    world_size : int, optional
        Number of training processes, by default 1
    """

    def __init__(
        self,
        data_dir: str,
        stats_dir: Optional[str] = None,
        channels: Optional[List[int]] = cfg().platform.channels,
        batch_size: int = cfg().platform.steps_per_batch,
        num_steps: int = cfg().platform.num_steps,
        stride: int = cfg().platform.stride,
        patch_size: Union[Tuple[int, int], int, None] = cfg().platform.patch_size,
        num_samples_per_year: Optional[int] = cfg().platform.num_samples_per_year,
        shuffle: bool = cfg().platform.shuffle,
        num_workers: int = cfg().task.num_workers,
        device: Union[str, torch.device] = cfg().task.device,
        process_rank: int = cfg().platform.process_rank,
        world_size: int = cfg().platform.world_size,
    ):
        super().__init__(meta=pmeta)
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.shuffle: bool = shuffle
        self.data_dir: Path = Path(data_dir)
        self.stats_dir: Path = Path(stats_dir) if stats_dir is not None else None
        self.channels: Optional[List[int]] = channels
        self.stride: int = stride
        self.num_steps: int = num_steps
        self.num_samples_per_year: int = num_samples_per_year
        self.process_rank: int = process_rank
        self.world_size: int = world_size

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size: Tuple[int, int] = patch_size

        # Set up device, needed for pipeline
        if isinstance(device, str):
            device = torch.device(device)
        # Need a index id if cuda
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda:0")
        self.device = device

        # check root directory exists
        if not self.data_dir.is_dir():
            raise IOError(f"Error, data directory {self.data_dir} does not exist")
        if self.stats_dir is not None and not self.stats_dir.is_dir():
            raise IOError(f"Error, stats directory {self.stats_dir} does not exist")

        self.parse_dataset_files()
        self.load_statistics()

        self.pipe = self._create_pipeline()

    def parse_dataset_files(self) -> None:
        """Parses the data directory for valid NC files and determines training samples

        Raises
        ------
        ValueError
            In channels specified or number of samples per year is not valid
        """
        # get all input data files
        self.data_paths = sorted(self.data_dir.glob("????.h5"))
        for data_path in self.data_paths:
            self.logger.info(f"MERRA2 file found: {data_path}")
        self.n_years = len(self.data_paths)
        self.logger.info(f"Number of years: {self.n_years}")

        # get total number of examples and image shape from the first file,
        # assuming other files have exactly the same format.
        self.logger.info(f"Getting file stats from {self.data_paths[0]}")
        with h5py.File(self.data_paths[0], "r") as f:
            # truncate the dataset to avoid out-of-range sampling
            data_samples_per_year = f["fields"].shape[0] - self.num_steps * self.stride
            self.img_shape = f["fields"].shape[2:]

            # If channels not provided, use all of them
            if self.channels is None:
                self.channels = [i for i in range(f["fields"].shape[1])]

            # If num_samples_per_year use all
            if self.num_samples_per_year is None:
                self.num_samples_per_year = data_samples_per_year

            # Adjust image shape if patch_size defined
            if self.patch_size is not None:
                self.img_shape = [
                    s - s % self.patch_size[i] for i, s in enumerate(self.img_shape)
                ]
            self.logger.info(f"Input image shape: {self.img_shape}")

            # Get total length
            self.total_length = self.n_years * self.num_samples_per_year
            self.length = self.total_length

            # Sanity checks
            if max(self.channels) >= f["fields"].shape[1]:
                raise ValueError(
                    f"Provided channel has indexes greater than the number \
                of fields {f['fields'].shape[1]}"
                )

            if self.num_samples_per_year > data_samples_per_year:
                raise ValueError(
                    f"num_samples_per_year ({self.num_samples_per_year}) > number of \
                    samples available ({data_samples_per_year})!"
                )

            self.logger.info(f"Number of samples/year: {self.num_samples_per_year}")
            self.logger.info(f"Number of channels available: {f['fields'].shape[1]}")

    def load_statistics(self) -> None:
        """Loads MERRA2 statistics from pre-computed numpy files

        The statistic files should be of name global_means.npy and global_std.npy with
        a shape of [1, C, 1, 1] located in the stat_dir.

        Raises
        ------
        IOError
            If mean or std numpy files are not found
        AssertionError
            If loaded numpy arrays are not of correct size
        """
        # If no stats dir we just skip loading the stats
        if self.stats_dir is None:
            self.mu = None
            self.std = None
            return
        # load normalisation values
        mean_stat_file = self.stats_dir / Path("global_means.npy")
        std_stat_file = self.stats_dir / Path("global_stds.npy")

        if not mean_stat_file.exists():
            raise IOError(f"Mean statistics file {mean_stat_file} not found")
        if not std_stat_file.exists():
            raise IOError(f"Std statistics file {std_stat_file} not found")

        # has shape [1, C, 1, 1]
        self.mu = np.load(str(mean_stat_file))[:, self.channels]
        # has shape [1, C, 1, 1]
        self.sd = np.load(str(std_stat_file))[:, self.channels]

        if not self.mu.shape == self.sd.shape == (1, len(self.channels), 1, 1):
            raise AssertionError("Error, normalisation arrays have wrong shape")

    def _create_pipeline(self) -> dali.Pipeline:
        """Create DALI pipeline

        Returns
        -------
        dali.Pipeline
            NC DALI pipeline
        """
        pipe = dali.Pipeline(
            batch_size=self.batch_size,
            num_threads=2,
            prefetch_queue_depth=2,
            py_num_workers=self.num_workers,
            device_id=self.device.index,
            py_start_method="spawn",
        )

        with pipe:
            source = MERRA2DaliExternalSource(
                data_paths=self.data_paths,
                num_samples=self.total_length,
                channels=self.channels,
                stride=self.stride,
                num_steps=self.num_steps,
                num_samples_per_year=self.num_samples_per_year,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                process_rank=self.process_rank,
                world_size=self.world_size,
            )
            # Update length of dataset
            self.length = len(source) // self.batch_size
            # Read current batch.
            invar, outvar = dali.fn.external_source(
                source,
                num_outputs=2,
                parallel=True,
                batch=False,
            )
            if self.device.type == "cuda":
                # Move tensors to GPU as external_source won't do that.
                invar = invar.gpu()
                outvar = outvar.gpu()

            # Crop.
            h, w = self.img_shape
            invar = invar[:, :h, :w]
            outvar = outvar[:, :, :h, :w]
            # Standardize.
            if self.stats_dir is not None:
                invar = dali.fn.normalize(invar, mean=self.mu[0], stddev=self.sd[0])
                outvar = dali.fn.normalize(outvar, mean=self.mu, stddev=self.sd)

            # Set outputs.
            pipe.set_outputs(invar, outvar)

        return pipe

    def __iter__(self):
        # Reset the pipeline before creating an iterator to enable epochs.
        self.pipe.reset()
        # Create DALI PyTorch iterator.
        return dali_pth.DALIGenericIterator([self.pipe], ["invar", "outvar"])

    def __len__(self):
        return self.length


class MERRA2DaliExternalSource:
    """DALI Source for lazy-loading the NC MERRA2 files

    Parameters
    ----------
    data_paths : Iterable[str]
        Directory where MERRA2 data is stored
    num_samples : int
        Total number of training samples
    channels : Iterable[int]
        List representing which MERRA2 variables to load
    stride : int
        Number of steps between input and output variables
    num_steps : int
        Number of timesteps are included in the output variables
    num_samples_per_year : int
        Number of samples randomly taken from each year
    batch_size : int, optional
        Batch size, by default 1
    shuffle : bool, optional
        Shuffle dataset, by default True
    process_rank : int, optional
        Rank ID of local process, by default 0
    world_size : int, optional
        Number of training processes, by default 1

    Note
    ----
    For more information about DALI external source operator:
    https://docs.nvidia.com/deeplearning/dali/archives/dali_1_13_0/user-guide/docs/examples/general/data_loading/parallel_external_source.html
    """

    def __init__(
        self,
        data_paths: Iterable[str],
        num_samples: int,
        channels: Iterable[int],
        num_steps: int,
        stride: int,
        num_samples_per_year: int,
        batch_size: int = 1,
        shuffle: bool = True,
        process_rank: int = 0,
        world_size: int = 1,
    ):
        self.data_paths = list(data_paths)
        # Will be populated later once each worker starts running in its own process.
        self.data_files = None
        self.num_samples = num_samples
        self.chans = list(channels)
        self.num_steps = num_steps
        self.stride = stride
        self.num_samples_per_year = num_samples_per_year
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.last_epoch = None

        self.indices = np.arange(num_samples)
        # Shard from indices if running in parallel
        self.indices = np.array_split(self.indices, world_size)[process_rank]

        # Get number of full batches, ignore possible last incomplete batch for now.
        # Also, DALI external source does not support incomplete batches in parallel mode.
        self.num_batches = len(self.indices) // self.batch_size

    def __call__(self, sample_info: dali.types.SampleInfo) -> Tuple[Tensor, Tensor]:
        if sample_info.iteration >= self.num_batches:
            raise StopIteration()

        if self.data_files is None:
            # This will be called once per worker. Workers are persistent,
            # so there is no need to explicitly close the files - this will be done
            # when corresponding pipeline/dataset is destroyed.
            self.data_files = [h5py.File(path, "r") for path in self.data_paths]

        # Shuffle before the next epoch starts.
        if self.shuffle and sample_info.epoch_idx != self.last_epoch:
            # All workers use the same rng seed so the resulting
            # indices are the same across workers.
            np.random.default_rng(seed=sample_info.epoch_idx).shuffle(self.indices)
            self.last_epoch = sample_info.epoch_idx

        # Get local indices from global index.
        idx = self.indices[sample_info.idx_in_epoch]
        year_idx = idx // self.num_samples_per_year
        in_idx = idx % self.num_samples_per_year

        train_steps = cfg().task.train_steps
        dts = cfg().task.data_timestep
        target_lead_times = [f"{iS * dts}h" for iS in range(1, train_steps + 1)]
        train_dates = year_range(*cfg().task.year_range, randomize=True)
        nepochs = cfg().task.nepoch
        max_iter = cfg().task.max_iter
        fmbatch: FMBatch = FMBatch(BatchType.Training)
        norms: Dict[str, xa.Dataset] = fmbatch.norm_data
        fmbatch.load_batch(forecast_date)
        train_data: xa.Dataset = fmbatch.get_train_data(day_offset)
        itf = batch.extract_inputs_targets_forcings(train_data, target_lead_times=target_lead_times, **dataclasses.asdict(cfg().task))
        train_inputs, train_targets, train_forcings = itf

        # data = self.data_files[year_idx]["fields"]
        # # Has [C,H,W] shape.
        # invar = data[in_idx, self.chans]
        #
        # # Has [T,C,H,W] shape.
        # outvar = np.empty((self.num_steps,) + invar.shape, dtype=invar.dtype)
        #
        # for i in range(self.num_steps):
        #     out_idx = in_idx + (i + 1) * self.stride
        #     outvar[i] = data[out_idx, self.chans]

        return train_inputs, train_targets

    def __len__(self):
        return len(self.indices)


class MERRA2InputIterator(object):
    def __init__(self):
        self.train_steps = cfg().task.train_steps
        self.dts = cfg().task.data_timestep
        self.n_day_offsets = 24//self.dts
        self.target_lead_times = [f"{iS * self.dts}h" for iS in self.train_steps]
        self.train_dates = year_range(*cfg().task.year_range, randomize=True)
        self.nepochs = cfg().task.nepoch
        self.max_iter = cfg().task.max_iter
        self.fmbatch: FMBatch = FMBatch(BatchType.Training)
        self.norms: Dict[str, xa.Dataset] = self.fmbatch.norm_data
        self.current_date = date(0,0,0 )

    def __iter__(self):
        self.i = 0
        self.n = len(self.train_dates)*self.n_day_offsets
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
        (inputs, targets, forcings) = self.extract_inputs_targets_forcings(train_data, target_lead_times=self.target_lead_times, **dataclasses.asdict(cfg().task))
        self.i = (self.i + 1) % self.n
        return inputs, targets, forcings
