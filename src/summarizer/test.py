import torch
from torch.utils import data
from transformers import AutoTokenizer

from .batch import SummarizerBatch
from .dataset import SummarizerDataset

from ..util.metric import accuracy
from ..autoencoder.model import AutoEncoder, AutoEncoderConfig
from ..util.prepare import prepare_random, prepare_device


def main():
	prepare_random()
	device = prepare_device()

	# rouge = evaluate.load("rouge")
	model_name = "google/flan-t5-small"
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	config = AutoEncoderConfig.from_pretrained(model_name, decoder_visibility=0)
	model: AutoEncoder = AutoEncoder.from_pretrained(model_name, config=config)

	checkpoint = "summarizer_01.pth"
	checkpoint = torch.load(checkpoint, map_location=device, weights_only=True)
	checkpoint = checkpoint["model"]

	model.load_state_dict(checkpoint)
	model.eval()
	model.to(device)

	dataset = SummarizerDataset(
		split="train",
		model_name=model_name,
		bos_token=config.pad_token_id,
	)
	dataloader = data.DataLoader(
		dataset=dataset,
		batch_size=24,
		shuffle=False,
		collate_fn=SummarizerBatch.collate_fn(
			pad_token=config.pad_token_id,
			ign_token=config.ign_token_id,
		),
	)

	for batch in dataloader:
		batch: SummarizerBatch = batch.to(device)

		outputs = model.generate(
			input_ids=batch.article_tokens,
			attention_mask=batch.article_padding,

			max_new_tokens=128,
			eos_token_id=model.eos_token,
			use_cache=False,

			do_sample=False,
			num_beams=1,
		)

		summary_masks = batch.summary_tokens != model.ign_token
		summaries = [
			summary[mask].tolist()
			for summary, mask in zip(batch.summary_tokens, summary_masks)
		]

		inputs = tokenizer.batch_decode(batch.article_tokens, skip_special_tokens=True, clean_up_tokenization_tokens=True)
		summaries = tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=True)
		outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

		for sample, summary, output in zip(inputs, summaries, outputs):
			print("INP:", sample)
			print("SUM:", summary)
			print("OUT:", output)
			print()
		break


if __name__ == "__main__":
	main()
