from functools import partial
from typing import List, Dict, Tuple, Callable

import datasets
import numpy as np
import torch

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizer

from .encoder import Encoder
from .model import AutoEncoderConfig, AutoEncoder
from ..util.prepare import prepare_random, prepare_device

EncodeState = Tuple[PreTrainedTokenizer, Encoder, torch.device]
encode_state: EncodeState | None = None

TokenizeState = Tuple[PreTrainedTokenizer,]
tokenize_state: TokenizeState | None = None


def encode_init(model_name: str, checkpoint: str) -> EncodeState:
	prepare_random()
	device = prepare_device()

	config = AutoEncoderConfig.from_pretrained(model_name)
	tokenizer = AutoTokenizer.from_pretrained(model_name)

	model = AutoEncoder.from_pretrained(model_name, config=config)
	model = model.encoder

	weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
	weights = {
		k.removeprefix("encoder."): v
		for k, v in weights["model"].items()
		if k.startswith("encoder.")
	}

	model.load_state_dict(weights)
	model.eval().to(device)

	return tokenizer, model, device


def encode(
	init_fn: Callable[[], EncodeState],
	dst_column: str,
	txt_column: List[str],
) -> Dict[str, np.array]:
	global encode_state
	torch.set_grad_enabled(False)

	encode_state = encode_state or init_fn()
	tokenizer, model, device = encode_state
	inputs = tokenizer(
		text=txt_column,
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
	embeds = embeds[mask].astype(np.float32)
	lengths = mask.sum(axis=1)

	if len(lengths) > 1:
		splits = np.cumsum(lengths[:-1])
		embeds = np.split(embeds, splits)
	else:
		embeds = [embeds]

	return {
		dst_column: embeds,
	}


def encode_dataset(
	dataset: Dataset | DatasetDict,
	model_name: str,
	checkpoint: str,
	src_column: str,
	dst_column: str,
	batch_size: int,
) -> Dataset | DatasetDict:
	dataset = dataset.map(
		function=partial(
			encode,
			partial(encode_init, model_name, checkpoint),
			dst_column,
		),
		input_columns=[src_column],
		batched=True,
		batch_size=batch_size,
		num_proc=1,
	)
	return dataset


def tokenize_init(tokenizer: str) -> TokenizeState:
	tokenizer = AutoTokenizer.from_pretrained(tokenizer)
	return tokenizer,


def tokenize(
	init_fn: Callable[[], TokenizeState],
	dst_column: List[str],
	txt_column: List[List[str]],
):
	global tokenize_state
	tokenize_state = tokenize_state or init_fn()

	tokenizer, = tokenize_state
	result = {
		dst_column: tokenizer(
			text=txt_column,
			padding=False,
			truncation=True,
			add_special_tokens=True,
			return_attention_mask=False,
		)["input_ids"]
	}

	return result


def tokenize_dataset(
	dataset: Dataset | DatasetDict,
	model_name: str,
	src_column: str,
	dst_column: str,
) -> Dataset | DatasetDict:
	dataset = dataset.map(
		function=partial(
			tokenize,
			partial(tokenize_init, model_name),
			dst_column,
		),
		input_columns=[src_column],
		batched=True,
	)
	return dataset


def main():
	model_name = "google/flan-t5-small"
	dataset = datasets.load_dataset(
		path="abisee/cnn_dailymail",
		name="3.0.0",
	)
	dataset = dataset["train"].select(range(10000))
	dataset = tokenize_dataset(
		dataset=dataset,
		model_name=model_name,
		src_column="highlights",
		dst_column="highlights_embeds",
	)
	dataset = encode_dataset(
		dataset=dataset,
		model_name=model_name,
		checkpoint="checkpoint/65586150dd2ce3eba3172ea837b748286e277200/autoencoder_00_34253.pth",
		src_column="article",
		dst_column="article_embeds",
		batch_size=16,
	)
	dataset.save_to_disk(
		"embeddings/cnndm1",
		max_shard_size="1GB"
	)


if __name__ == "__main__":
	main()
