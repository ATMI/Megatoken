import torch
from tqdm import tqdm

import prepare
from metric import accuracy, RollingMean
from model import Model


def main():
	torch.set_grad_enabled(False)
	torch.autograd.detect_anomaly(True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	_, test_loader  = prepare.dataloaders()

	checkpoint = torch.load("checkpoint/20312.pth", map_location=device, weights_only=True)

	model = Model()
	# model.eval()
	model.load_state_dict(checkpoint["model"])
	model = model.to(device)

	bar = tqdm(total=len(test_loader))
	rolling = RollingMean(100)

	for batch in test_loader:
		batch = batch.to(device)

		dense, filled = model.forward(
			batch.tokens, batch.pad_mask,
			batch.sparse, batch.pad_mask,
		)

		acc = accuracy(filled, batch.labels) * 100
		acc_, = rolling(acc)

		bar.set_postfix(acc=acc_)
		bar.update(1)

	bar.close()


if __name__ == "__main__":
	main()
