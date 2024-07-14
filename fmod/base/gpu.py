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

def memory_snapshot_path( ) -> str:
	cpath = f"{cfg().platform.results}/memory/snapshot.{cfg().task.training_version}.pkl"
	os.makedirs(os.path.dirname(cpath), 0o777, exist_ok=True)
	return cpath

def save_memory_snapshot():
	if cfg().pipeline.memory_debug:
		cuda.memory._dump_snapshot( memory_snapshot_path() )
