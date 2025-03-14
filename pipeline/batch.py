from abc import abstractmethod


class Batch:
	def __init__(self, batch):
		self.x = Batch.collate_x(batch)
		self.y = Batch.collate_y(batch)

	@staticmethod
	@abstractmethod
	def collate_x(batch):
		pass

	@staticmethod
	@abstractmethod
	def collate_y(batch):
		pass
