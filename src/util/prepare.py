import random
import torch


def prepare_random():
	# torch.backends.cudnn.deterministic = True
	# torch.backends.cudnn.benchmark = False
	seed = 42
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)


def prepare_device():
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")
