from torch.utils import data
from transformers import BertTokenizer
import datasets
from src.autoencoder.config import Config
from src.util import prepare
from src.util.tensorfile import TensorReader


class Dataset(data.Dataset):
	def __init__(self, split: str):
		super(Dataset, self).__init__()

		self.embed = TensorReader(f"bert-embeds.{split}")
		self.ds = prepare.dataset()[split]

	def __len__(self):
		return len(self.ds)

	def __getitem__(self, idx):
		sample = self.ds[idx]
		embed = self.embed.by_id(sample["id"])
		label = sample["label"] > 2
		return embed, label




def main(model_name):
	def tokenize(batch, idx):
		tokenizer = BertTokenizer.from_pretrained(model_name)
		tokens = tokenizer(
			batch["text"],
			padding=False,
			truncation=True,
			max_length=Config.max_length,
			return_attention_mask=False,
		)
		tokens = tokens["input_ids"]
		return {
			"idx": idx,
			"tokens": tokens,
		}

	ds = datasets.load_dataset(Config.dataset)
	ds = ds.map(tokenize, with_indices=True, batched=True, num_proc=4)
	ds.save_to_disk("dataset-bert")


if __name__ == "__main__":
	ds = datasets.load_from_disk("dataset-bert")
