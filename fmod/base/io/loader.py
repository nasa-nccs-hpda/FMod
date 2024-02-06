from torch.utils.data.dataset import IterableDataset

class BaseDataset(IterableDataset):

	def __init__(self, lenght: int ):
		self.lenght = lenght

	def __len__(self):
		return self.lenght