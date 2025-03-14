from abc import abstractmethod


class Batch:
	@abstractmethod
	def __init__(self, batch):
		pass

	@abstractmethod
	def x(self):
		pass

	@abstractmethod
	def y(self):
		pass
