from pathlib import Path

import datasets
from torch import nn
from torch.utils import data
from transformers import AutoTokenizer

from config import Config
from embedding import Embedding
from encoder import Encoder
from decoder import Decoder


def dataset():
	path = Path("dataset")
	if path.exists():
		return datasets.load_from_disk(path)

	def tokenize(batch):
		tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer)
		tokens = tokenizer(
			batch["text"],
			padding=False,
			truncation=True,
			max_length=Config.max_length,
			return_attention_mask=False,
		)
		tokens = tokens["input_ids"]
		return {"tokens": tokens}

	ds = datasets.load_dataset(Config.dataset)
	ds = ds.map(tokenize, batched=True, num_proc=4)
	ds = ds.select_columns(["tokens"])
	ds.save_to_disk(path)

	return ds


def dataloaders(collate_batch):
	ds = dataset()
	train_loader = data.DataLoader(
		dataset=ds["train"],
		batch_size=Config.batch_size,
		shuffle=True,
		collate_fn=collate_batch,
	)
	test_loader = data.DataLoader(
		dataset=ds["test"],
		batch_size=Config.batch_size,
		shuffle=False,
		collate_fn=collate_batch,
	)

	return train_loader, test_loader


def embedding():
	return Embedding(
		model_dim=Config.model_dim,
		vocab_size=Config.vocab_size,
		pad_token=Config.pad_token,
		max_len=Config.max_length,
	)


def encoder():
	return Encoder(
		model_dim=Config.model_dim,
		head_num=Config.head_num,
		fc_dim=Config.fc_dim,
		activation=Config.activation,
		layer_num=Config.encoder_layers,

		bias=Config.bias,
		temperature=Config.temperature,
	)


def decoder():
	return Decoder(
		vocab_size=Config.vocab_size,
		model_dim=Config.model_dim,
		head_num=Config.head_num,
		fc_dim=Config.fc_dim,
		activation=Config.activation,
		layer_num=Config.decoder_layers,
	)
