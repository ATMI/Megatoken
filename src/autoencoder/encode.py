import argparse
from typing import Callable, List

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

from src.autoencoder.dataset import AutoEncoderDataset
from src.autoencoder.encoder import Encoder
from src.autoencoder.model import AutoEncoderConfig, AutoEncoder
from src.util.prepare import prepare_random, prepare_device
from src.util.tensorfile import TensorWriter


def encode_dataset(dataset: Dataset, checkpoint: str) -> Callable[[List[str]], List[np.array]]:
	prepare_random()
	device = prepare_device()
	torch.set_grad_enabled(False)

	model_name = "google/flan-t5-small"

	model: Encoder | None = None

	def init_worker():
		nonlocal model, checkpoint

		config = AutoEncoderConfig.from_pretrained(model_name)
		model: AutoEncoder = AutoEncoder.from_pretrained(model_name, config=config)
		model: Encoder = model.encoder

		weights = torch.load(checkpoint, map_location=device, weights_only=True)
		weights = {
			k.removeprefix("encoder."): v
			for k, v in weights["model"].items()
			if k.startswith("encoder.")
		}

		model.load_state_dict(weights)
		model.eval().to(device)

	def encode(text: List[str]) -> List[np.array]:
		pass


def main():
	args = argparse.ArgumentParser()
	args.add_argument("checkpoint", help="Checkpoint file")
	args.add_argument("subset", help="Subset to use: train or test")
	args.add_argument("output", help="Output file")
	args = args.parse_args()

	# TODO: replace the arguments, move similar logic to separate functions
	dataset = AutoEncoderDataset(
		name="abisee/cnn_dailymail",
		version="3.0.0",
		split=args.subset,
		text_column="article",
		tokenizer=model_name,
		ign_token=config.ign_token_id,
	)
	dataloader = data.DataLoader(
		dataset=dataset,
		batch_size=16,
		shuffle=False,
		collate_fn=EncoderBatch.collate_fn(
			pad_token=config.pad_token_id,
		),
	)

	writer = TensorWriter(args.output)
	for batch in tqdm(dataloader):
		batch: EncoderBatch = batch.to(device)
		output: Encoder.Output = model(
			input_ids=batch.input_ids,
			attention_mask=batch.pad_mask,
		)

		prune_mask = output.prune_masks[:, -1]
		mask = (prune_mask > -1) & batch.pad_mask

		embeds = output.last_hidden_state
		embeds = embeds[mask]
		lengths = mask.sum(dim=1)

		embeds = embeds.cpu()
		lengths = lengths.tolist()

		head = 0
		for idx, length in enumerate(lengths):
			tail = head + length
			embed = embeds[head:tail]
			head = tail
			writer.write(idx, embed)
	writer.close()


if __name__ == "__main__":
	main()
