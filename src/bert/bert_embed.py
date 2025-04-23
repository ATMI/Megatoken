import argparse
import os

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import datasets

from ..util import prepare
from ..util.tensorfile import TensorWriter
from ..autoencoder.config import Config
from .dataset import main as tokenize_dataset

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def collate_fn(batch):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	ids = [sample["idx"] for sample in batch]

	data = [torch.tensor(sample["tokens"]) for sample in batch]
	padded_data = pad_sequence(data, batch_first=True, padding_value=tokenizer.pad_token_id)
	attention_masks = (padded_data != tokenizer.pad_token_id)

	x = {
		"input_ids": padded_data.to(device),
		"attention_mask": attention_masks.to(device),
	}
	return x, ids


def main():
	args = argparse.ArgumentParser()
	# args.add_argument("name", help="Bert model name")
	args.add_argument("subset", help="Subset to use: train or test")
	args = args.parse_args()

	name = "bert-base-uncased"

	torch.set_grad_enabled(False)
	prepare.rnd(Config.seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if not os.path.exists('dataset-bert'):
		tokenize_dataset(name)
	dataset = datasets.load_from_disk("dataset-bert")

	dataloader = data.DataLoader(
		dataset=dataset[args.subset],
		batch_size=Config.batch_size,
		collate_fn=collate_fn,
		shuffle=False,
	)

	model = BertModel.from_pretrained(name)
	model = model.eval()
	model = model.to(device)

	writer = TensorWriter(f"bert-embeds.{args.subset}")
	with torch.no_grad():
		for x, ids in tqdm(dataloader):
			out = model.forward(**x)

			embeds = out.pooler_output
			embeds = embeds.cpu()

			for i, id in enumerate(ids):
				part = embeds[i, :]
				writer.write(id, part)

	writer.close()


if __name__ == "__main__":
	main()
