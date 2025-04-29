from functools import partial
from typing import List, Dict, Tuple, Callable

import datasets
import numpy as np
import torch

from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from .encoder import Encoder
from .model import AutoEncoderConfig, AutoEncoder
from ..util.prepare import prepare_random, prepare_device

WorkerState = Tuple[PreTrainedTokenizer, Encoder, torch.device]
state: WorkerState | None = None


def init(model_name: str, checkpoint: str) -> WorkerState:
	prepare_random()
	device = prepare_device()

	config = AutoEncoderConfig.from_pretrained(model_name)
	tokenizer = AutoTokenizer.from_pretrained(model_name)

	model = AutoEncoder.from_pretrained(model_name, config=config)
	model = model.encoder

	weights = torch.load(checkpoint, map_location=device, weights_only=True)
	weights = {
		k.removeprefix("encoder."): v
		for k, v in weights["model"].items()
		if k.startswith("encoder.")
	}

	model.load_state_dict(weights)
	model.eval().to(device)

	return tokenizer, model, device


def encode(
	init_fn: Callable[[], WorkerState],
	dst_column: str,
	text: List[str],
) -> Dict[str, np.array]:
	global state
	torch.set_grad_enabled(False)

	state = state or init_fn()
	tokenizer, model, device = state
	inputs = tokenizer(
		text=text,
		padding=True,
		truncation=True,
		add_special_tokens=True,

		return_tensors="pt",
		return_attention_mask=True,
	)

	outputs: Encoder.Output = model(
		input_ids=inputs.input_ids.to(device),
		attention_mask=inputs.attention_mask.to(device),
	)

	mask = outputs.prune_masks[:, -1].cpu().numpy()
	embeds = outputs.last_hidden_state.cpu().numpy()
	attention_mask = inputs.attention_mask.numpy().astype(bool)

	del inputs, outputs

	mask = (mask > -1) & attention_mask
	embeds = embeds[mask]
	lengths = mask.sum(axis=1)

	if len(lengths) > 1:
		embeds = np.split(embeds, lengths[:-1])

	return {
		dst_column: embeds,
	}


def encode_dataset(
	dataset: Dataset,
	model_name: str,
	checkpoint: str,
	src_column: str,
	dst_column: str,
	batch_size: int,
) -> Dataset:
	dataset = dataset.map(
		function=partial(
			encode,
			partial(init, model_name, checkpoint),
			dst_column,
		),
		input_columns=[src_column],
		batched=True,
		batch_size=batch_size,
		num_proc=1,
	)
	return dataset


def main():
	dataset = datasets.load_dataset(
		path="abisee/cnn_dailymail",
		name="3.0.0",
	)
	dataset = encode_dataset(
		dataset=dataset,
		model_name="google/flan-t5-small",
		checkpoint="checkpoint/65586150dd2ce3eba3172ea837b748286e277200/autoencoder_00_34253.pth",
		src_column="article",
		dst_column="article_embeds",
		batch_size=8,
	)
	dataset.save_to_disk("embeddings/cnndm")


if __name__ == "__main__":
	main()
