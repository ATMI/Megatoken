import argparse

import torch
from torch.utils import data
from tqdm import tqdm

from .batch import Batch
from .config import Config
from .autoencoder import AutoEncoder
from ..util import prepare
from ..util.tensorfile import TensorWriter


def main():
	args = argparse.ArgumentParser()
	args.add_argument("checkpoint", help="Checkpoint file")
	args.add_argument("subset", help="Subset to use: train or test")
	args = args.parse_args()

	torch.set_grad_enabled(False)
	prepare.prepare_random(Config.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	dataset = prepare.dataset()
	dataloader = data.DataLoader(
		dataset=dataset[args.subset],
		batch_size=Config.batch_size,
		collate_fn=Batch.collate,
	)

	model = AutoEncoder(Config.model, Config.bias, Config.temperature)
	model = model.eval()
	model = model.to(device)

	init = torch.load(args.checkpoint, map_location=device, weights_only=True)
	model.load_state_dict(init["model"])

	writer = TensorWriter(f"embeds.{args.subset}")
	for batch in tqdm(dataloader):
		batch = batch.to(device)
		result = model.encode(
			src_tokens=batch.tokens,
			eos_mask=batch.eos_mask,
			pad_mask=batch.pad_mask,
			attn_mask=None,
			attn_scores=False,
		)

		mask = (result.gate_mask > -1) & result.pad_mask
		embeds = result.input_embeds[mask]
		mask = mask.sum(dim=1)

		embeds = embeds.cpu()
		mask = mask.cpu()

		head = 0
		for id, length in zip(batch.ids, mask):
			tail = head + length
			part = embeds[head:tail]
			head = tail
			writer.write(id, part)
	writer.close()


if __name__ == "__main__":
	main()
