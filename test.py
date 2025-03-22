import torch

import prepare
from model import Model


def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	_, test_loader = prepare.dataloaders()

	checkpoint = torch.load("checkpoint/20312.pth", map_location=device, weights_only=True)

	model = Model()
	model.load_state_dict(checkpoint["model"])
	model = model.to(device)

	for batch in test_loader:
		batch = batch.to(device)


if __name__ == "__main__":
	main()
