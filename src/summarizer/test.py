import datasets
import evaluate
import torch
from torch.utils import data
from transformers import AutoTokenizer

from .batch import Batch
from .dataset import Dataset
from ..autoencoder.config import Config
from ..autoencoder.autoencoder import AutoEncoder
from ..util import prepare

text = "(CNN) -- Authorities have seized more than 1,000 pirate costumes made in China and destined for sale in Washington state because they contained high levels of lead, officials said. Shipments worth $10,000 were on the way to a distributor in Seattle when they were seized by U.S. customs officials. The Consumer Product Safety Commission found the costumes contained more than 11 times the allowable level of lead. Officials did not specify when the seizures occurred but said the tainted products will be destroyed."


def main():
	prepare.prepare_random(Config.seed)
	torch.set_grad_enabled(False)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = AutoEncoder(Config.model, Config.bias, Config.temperature)
	model.eval()
	# model = model.to(device)

	# init = torch.load("1.pth", weights_only=True, map_location=device)
	# model.t5.encoder.load_state_dict(init["t5"]["encoder"])
	#
	# init = torch.load("summarizer_1.pth", weights_only=True, map_location=device)
	# model.t5.decoder.load_state_dict(init["t5"]["decoder"])

	init = torch.load("summarizer_2.pth", weights_only=True)
	init = init["model"]

	del model.t5.encoder
	model.load_state_dict(init, strict=False)
	model = model.to(device)

	tokenizer = AutoTokenizer.from_pretrained(Config.model)
	rouge = evaluate.load("rouge")

	# source = tokenizer(
	# 	text,
	# 	padding=False,
	# 	truncation=True,
	# 	max_length=Config.max_length,
	# 	return_tensors="pt",
	# )

	# source = source["input_ids"].to(device)

	# memory = model.encode(
	# 	tokens=source,
	# 	eos_mask=torch.tensor([[0], [len(source) - 1]], device=device),
	# 	pad_mask=torch.ones_like(source, dtype=torch.bool),
	# 	attn_mask=None,
	# 	attn_scores=False,
	# )

	dataset = datasets.load_from_disk("dataset")
	dataset = dataset["test"]
	batch_size = 2

	dataset = Dataset("embeds.test", dataset)
	dataloader = data.DataLoader(
		dataset,
		batch_size=batch_size,
		collate_fn=Batch.collate,
	)

	dataloader = iter(dataloader)
	scores = []

	for i in range(100):
		batch = next(dataloader)
		batch = batch.to(device)

		indices = torch.arange(batch_size)
		target = torch.full((batch_size, 1), Config.pad_token, device=device)
		result = [None] * batch_size

		for _ in range(Config.max_length):
			logits = model.decode(
				memory=batch.memory,
				tokens=target,
				pad_mask=None,
				attn_mask=None,
			)

			logits = logits[:, -1]
			tokens = logits.argmax(dim=1)

			mask = []
			for idx, token in enumerate(tokens):
				if token == Config.eos_token:
					idx_ = indices[idx].item()
					result[idx_] = target[idx].tolist()
				else:
					mask.append(idx)

			if not mask:
				target = []
				break

			if len(mask) != len(tokens):
				indices = indices[mask]
				target = target[mask]
				tokens = tokens[mask]

				memory = batch.memory
				memory = AutoEncoder.Memory(
					embeds=memory.input_embeds[mask],
					pad_mask=memory.pad_mask[mask],
					kv_dim=None,
					gate_masks=None,
					attn_scores=None,
				)
				batch.memory = memory

			tokens = tokens.unsqueeze(1)
			target = torch.cat((target, tokens), dim=1)

		for idx, target in enumerate(target):
			idx = indices[idx]
			result[idx] = target[idx].tolist()

		result = tokenizer.batch_decode(result, skip_special_tokens=True)
		target = batch.text

		result = rouge.compute(
			predictions=result,
			references=target,
			rouge_types=["rouge1", "rouge2", "rougeL"],
			use_stemmer=True,
		)

		result = result["rouge1"]
		scores.append(result)

	print(sum(scores) / len(scores))


if __name__ == "__main__":
	main()
