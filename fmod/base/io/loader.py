from torch.utils.data.dataset import IterableDataset
from typing import Any, Dict, List, Tuple, Type, Optional, Union
# from fmod.pipeline.merra2 import TensorRole

class BaseDataset(IterableDataset):

	def __init__(self, lenght: int ):
		super(BaseDataset, self).__init__()
		self.length = lenght
		self.chanIds: Dict[str,List[str]] = {}

	def channel_ids(self, role: str) -> List[str]:
		return self.chanIds[role]

	def __len__(self):
		return self.length