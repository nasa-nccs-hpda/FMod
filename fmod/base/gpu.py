from fmod.base.util.config import cfg
from torch import cuda
import torch, os

def set_device() -> torch.device:
	gpu_index = cfg().pipeline.gpu
	device = torch.device(f'cuda:{gpu_index}' if cuda.is_available() else 'cpu')
	if cuda.is_available():
		cuda.set_device(device.index)
		if cfg().pipeline.memory_debug:
			cuda.memory._record_memory_history()
	else:
		assert gpu_index == 0, "Can't run on multiple GPUs: No GPUs available"
	return device


def get_device() -> torch.device:
	gpu_index = cfg().pipeline.gpu
	device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
	return device

def save_memory_snapshot():
	if cfg().pipeline.memory_debug:
		ssdir = f"{cfg().dataset.dataset_root}/cuda/memory_snapshots"
		os.makedirs(ssdir, exist_ok=True)
		ssfile = f"{ssdir}/{cfg().task.training_version}.pkl"
		cuda.memory._dump_snapshot(ssfile)
