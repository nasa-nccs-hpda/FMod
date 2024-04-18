# ruff: noqa: E402
import torch, os, json
os.environ["TORCHELASTIC_ENABLE_FILE_TIMER"] = "1"
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig, ListConfig
from modulus.distributed import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.utils.generative import EasyDict
from models.sres.corrdiff.datasets.dataset import init_dataset_from_config


valid_archs = {"ddpmpp-cwb", "ddpmpp-cwb-v0-regression", "ncsnpp", "adm"}

def process_config(config: DictConfig, dist: DistributedManager) -> EasyDict:
	if not hasattr(config, "task"):
		raise ValueError(
			"""Need to specify the task. Make sure the right config file is used. Run training using python train.py --config-name=<your_yaml_file>.
			For example, for regression training, run python train.py --config-name=config_train_regression.
			And for diffusion training, run python train.py --config-name=config_train_diffusion."""
		)

	# Dump the configs
	os.makedirs(config.outdir, exist_ok=True)
	OmegaConf.save(config, os.path.join(config.outdir, "config.yaml"))

	# Parse options
	regression_checkpoint_path = getattr(config, "regression_checkpoint_path", None)
	if regression_checkpoint_path:
		regression_checkpoint_path = to_absolute_path(regression_checkpoint_path)
	task = getattr(config, "task")
	outdir = getattr(config, "outdir", "./output")
	arch = getattr(config, "arch", "ddpmpp-cwb-v0-regression")
	precond = getattr(config, "precond", "unetregression")

	# parse hyperparameters
	duration = getattr(config, "duration", 200)
	batch_size_global = getattr(config, "batch_size_global", 256)
	batch_size_gpu = getattr(config, "batch_size_gpu", 2)
	cbase = getattr(config, "cbase", 1)
	# cres = parse_int_list(getattr(config, "cres", None))
	lr = getattr(config, "lr", 0.0002)
	ema = getattr(config, "ema", 0.5)
	dropout = getattr(config, "dropout", 0.13)
	augment = getattr(config, "augment", 0.0)

	# Parse performance options
	fp16 = getattr(config, "fp16", False)
	ls = getattr(config, "ls", 1)
	bench = getattr(config, "bench", True)
	workers = getattr(config, "workers", 4)

	# Parse I/O-related options
	wandb_mode = getattr(config, "wandb_mode", "disabled")
	tick = getattr(config, "tick", 1)
	dump = getattr(config, "dump", 500)
	seed = getattr(config, "seed", 0)
	transfer = getattr(config, "transfer")
	dry_run = getattr(config, "dry_run", False)

	# Parse weather data options
	c = EasyDict()
	c.task = task
	c.wandb_mode = wandb_mode
	c.patch_shape_x = getattr(config, "patch_shape_x", None)
	c.patch_shape_y = getattr(config, "patch_shape_y", None)
	config.dataset.data_path = to_absolute_path(config.dataset.data_path)
	if hasattr(config, "global_means_path"):
		config.global_means_path = to_absolute_path(config.global_means_path)
	if hasattr(config, "global_stds_path"):
		config.global_stds_path = to_absolute_path(config.global_stds_path)
	dataset_config = OmegaConf.to_container(config.dataset)
	data_loader_kwargs = EasyDict( pin_memory=True, num_workers=workers, prefetch_factor=2 )

	# Initialize logger.
	os.makedirs("logs", exist_ok=True)
	logger = PythonLogger(name="train")  # General python logger
	logger0 = RankZeroLoggingWrapper(logger, dist)
	logger.file_logging(file_name=f"logs/train_{dist.rank}.log")

	# inform about the output
	logger.info( f"Checkpoints, logs, configs, and stats will be written in this directory: {os.getcwd()}" )

	# Initialize config dict.
	c.network_kwargs = EasyDict()
	c.loss_kwargs = EasyDict()
	c.optimizer_kwargs = EasyDict( class_name="torch.optim.Adam", lr=lr, betas=[0.9, 0.999], eps=1e-8 )

	# Network architecture.

	if arch not in valid_archs:
		raise ValueError( f"Invalid network architecture {arch}; " f"valid choices are {valid_archs}" )

	if arch == "ddpmpp-cwb":
		c.network_kwargs.update(
			model_type="SongUNet",
			embedding_type="positional",
			encoder_type="standard",
			decoder_type="standard",
		)  # , attn_resolutions=[28]
		c.network_kwargs.update(
			channel_mult_noise=1,
			resample_filter=[1, 1],
			model_channels=128,
			channel_mult=[1, 2, 2, 2, 2],
			attn_resolutions=[28],
		)  # era5-cwb, 448x448

	elif arch == "ddpmpp-cwb-v0-regression":
		c.network_kwargs.update(
			model_type="SongUNet",
			embedding_type="zero",
			encoder_type="standard",
			decoder_type="standard",
		)  # , attn_resolutions=[28]
		c.network_kwargs.update(
			channel_mult_noise=1,
			resample_filter=[1, 1],
			model_channels=128,
			channel_mult=[1, 2, 2, 2, 2],
			attn_resolutions=[28],
		)  # era5-cwb, 448x448

	elif arch == "ncsnpp":
		c.network_kwargs.update(
			model_type="SongUNet",
			embedding_type="fourier",
			encoder_type="residual",
			decoder_type="standard",
		)
		c.network_kwargs.update(
			channel_mult_noise=2,
			resample_filter=[1, 3, 3, 1],
			model_channels=128,
			channel_mult=[2, 2, 2],
		)

	else:
		c.network_kwargs.update( model_type="DhariwalUNet", model_channels=192, channel_mult=[1, 2, 3, 4] )

	# Preconditioning & loss function.
	if precond == "edmv2" or precond == "edm":
		c.network_kwargs.class_name = "training.networks.EDMPrecondSRV2"
		c.loss_kwargs.class_name = "modulus.metrics.diffusion.EDMLossSR"
	elif precond == "edmv1":
		c.network_kwargs.class_name = "training.networks.EDMPrecondSR"
		c.loss_kwargs.class_name = "modulus.metrics.diffusion.EDMLossSR"
	elif precond == "unetregression":
		c.network_kwargs.class_name = "modulus.models.diffusion.UNet"
		c.loss_kwargs.class_name = "modulus.metrics.diffusion.RegressionLoss"
	elif precond == "resloss":
		c.network_kwargs.class_name = "modulus.models.diffusion.EDMPrecondSR"
		c.loss_kwargs.class_name = "modulus.metrics.diffusion.ResLoss"

	# Network options.
	if cbase is not None:
		c.network_kwargs.model_channels = cbase
	# if cres is not None:
	#    c.network_kwargs.channel_mult = cres
	if augment > 0:
		raise NotImplementedError("Augmentation is not implemented")
	c.network_kwargs.update(dropout=dropout, use_fp16=fp16)

	# Training options.
	c.total_kimg = max(int(duration * 1000), 1)
	c.ema_halflife_kimg = int(ema * 1000)
	c.update(batch_size_gpu=batch_size_gpu, batch_size_global=batch_size_global)
	c.update(loss_scaling=ls, cudnn_benchmark=bench)
	c.update(kimg_per_tick=tick, state_dump_ticks=dump)
	if regression_checkpoint_path:
		c.regression_checkpoint_path = regression_checkpoint_path

	# Random seed.
	if seed is None:
		seed = torch.randint(1 << 31, size=[], device=dist.device)
		if dist.distributed:
			torch.distributed.broadcast(seed, src=0)
		seed = int(seed)

	# Transfer learning and resume.
	if transfer is not None:
		c.resume_pkl = transfer
		c.ema_rampup_ratio = None

	c.run_dir = outdir

	# Print options.
	for key in list(c.keys()):
		val = c[key]
		if isinstance(val, (ListConfig, DictConfig)):
			c[key] = OmegaConf.to_container(val, resolve=True)
	logger0.info("Training options:")
	logger0.info(json.dumps(c, indent=2))
	logger0.info(f"Output directory:        {c.run_dir}")
	logger0.info(f"Dataset path:            {dataset_config['data_path']}")
	logger0.info(f"Network architecture:    {arch}")
	logger0.info(f"Preconditioning & loss:  {precond}")
	logger0.info(f"Number of GPUs:          {dist.world_size}")
	logger0.info(f"Batch size:              {c.batch_size_global}")
	logger0.info(f"Mixed-precision:         {c.network_kwargs.use_fp16}")

	# Create output directory.
	logger0.info("Creating output directory...")
	if dist.rank == 0:
		os.makedirs(c.run_dir, exist_ok=True)
		with open(os.path.join(c.run_dir, "training_options.json"), "wt") as f:
			json.dump(c, f, indent=2)

	return c

	(dataset, dataset_iterator) = init_dataset_from_config(
		dataset_config, data_loader_kwargs, batch_size=batch_size_gpu, seed=seed
	)

	(img_shape_y, img_shape_x) = dataset.image_shape()
	if (c.patch_shape_x is None) or (c.patch_shape_x > img_shape_x):
		c.patch_shape_x = img_shape_x
	if (c.patch_shape_y is None) or (c.patch_shape_y > img_shape_y):
		c.patch_shape_y = img_shape_y
	if c.patch_shape_x != c.patch_shape_y:
		raise NotImplementedError("Rectangular patch not supported yet")
	if c.patch_shape_x % 32 != 0 or c.patch_shape_y % 32 != 0:
		raise ValueError("Patch shape needs to be a multiple of 32")
	if c.patch_shape_x != img_shape_x or c.patch_shape_y != img_shape_y:
		logger0.info("Patch-based training enabled")
	else:
		logger0.info("Patch-based training disabled")
