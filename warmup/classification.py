import datasets
import torch
from pathlib import Path
from torch import nn, optim
from torch.nn.utils import rnn
from torch.utils import data
from tqdm import tqdm
from transformers import AutoTokenizer


def load_tokenizer():
	return AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")


def tokenize_dataset(data):
	tokenizer = load_tokenizer()

	tokens = tokenizer(
		data["text"],
		return_attention_mask=False,
		return_token_type_ids=False,
		truncation=True,
	)
	tokens = tokens["input_ids"]

	return {
		"tokens": tokens,
	}


def prepare_dataset(path):
	dataset = datasets.load_dataset("stanfordnlp/imdb")
	dataset = dataset.map(tokenize_dataset, batched=True)
	dataset = dataset.remove_columns("text")
	dataset.save_to_disk(path)


class Model(nn.Module):
	def __init__(self, vocab_size):
		super().__init__()

		self.embedding = nn.Embedding(
			num_embeddings=vocab_size,
			embedding_dim=128,
		)
		self.encoder = nn.TransformerEncoder(
			num_layers=4,
			encoder_layer=nn.TransformerEncoderLayer(
				d_model=128,
				nhead=4,
				dim_feedforward=512,
				batch_first=True,
			),
		)
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(128, 1),
		)

	def forward(self, tokens, mask):
		x = self.embedding(tokens)
		x = self.encoder(x, src_key_padding_mask=mask)
		x = x[:, 0]
		x = self.classifier(x)
		return x


def epoch_pass(epoch, model, dataloader, train):
	if train:
		torch.set_grad_enabled(True)
		model.train()
	else:
		torch.set_grad_enabled(False)
		model.eval()

	epoch_loss = 0
	num_batches = 0
	num_correct = 0
	num_total = 0

	label = "Train" if train else "Test"
	pbar = tqdm(
		total=len(dataloader),
		desc=f"{label} {epoch + 1}/{num_epoch}",
	)

	for tokens, mask, labels in dataloader:
		tokens = tokens.to(device)
		labels = labels.to(device)
		mask = mask.to(device)

		if train:
			optimizer.zero_grad()

		outputs = model(tokens, mask)
		loss = criterion(outputs, labels)

		if train:
			loss.backward()
			optimizer.step()

		epoch_loss += loss.item()
		num_batches += 1
		num_correct += (outputs.sigmoid() > 0.5).eq(labels).sum().item()
		num_total += outputs.size(0)

		pbar.set_postfix({
			"Loss": f"{epoch_loss / num_batches:.4f}",
			"Accuracy": f"{100 * num_correct / num_total:.2f}",
		})
		pbar.update(1)

	pbar.close()


def collate_batch(batch, pad):
	tokens = rnn.pad_sequence(
		[torch.tensor(i["tokens"]) for i in batch],
		batch_first=True,
		padding_value=pad,
		padding_side="right",
	)

	batch_size, seq_length = tokens.shape
	pad_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool)

	for mask, row in zip(pad_mask, batch):
		mask[len(row["tokens"]):] = True

	labels = torch.tensor([i["label"] for i in batch], dtype=torch.float).unsqueeze(1)
	return tokens, pad_mask, labels


if __name__ == "__main__":
	dataset_path = "data/imdb"
	checkpoint_dir = Path("checkpoint")
	checkpoint_dir.mkdir(parents=True, exist_ok=True)

	if not checkpoint_dir.exists():
		prepare_dataset(dataset_path)
	dataset = datasets.load_from_disk(dataset_path)

	train_dataset = dataset["train"]
	test_dataset = dataset["test"]

	tokenizer = load_tokenizer()
	train_loader = data.DataLoader(
		train_dataset,
		batch_size=64,
		num_workers=2,
		collate_fn=lambda x: collate_batch(x, tokenizer.pad_token_id),
		shuffle=True,
	)
	test_loader = data.DataLoader(
		test_dataset,
		batch_size=128,
		num_workers=2,
		collate_fn=lambda x: collate_batch(x, tokenizer.pad_token_id),
		shuffle=False,
	)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = Model(tokenizer.vocab_size)
	model.train()
	model.to(device)

	optimizer = optim.Adam(model.parameters(), lr=0.0001)
	criterion = nn.BCEWithLogitsLoss()

	num_epoch = 10
	for epoch in range(num_epoch):
		epoch_pass(epoch, model, train_loader, True)
		epoch_pass(epoch, model, test_loader, False)

		checkpoint_path = checkpoint_dir / f"{epoch + 1}.pt"
		checkpoint = {
			"epoch": epoch,
			"model": model.state_dict(),
			"optimizer": optimizer.state_dict(),
		}
		torch.save(checkpoint, checkpoint_path)
