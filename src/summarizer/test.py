import evaluate
import torch
import datasets
from torch.utils import data
from tqdm import tqdm
from transformers import AutoTokenizer

from ..autoencoder.encoder import Encoder
from .batch import SummarizerBatch
from .dataset import SummarizerDataset
from ..autoencoder.model import AutoEncoder, AutoEncoderConfig
from ..util.prepare import prepare_random, prepare_device


def main():
	prepare_random()
	device = prepare_device()

	rouge = evaluate.load("rouge")
	model_name = "google/flan-t5-small"
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	config = AutoEncoderConfig.from_pretrained(model_name, decoder_visibility=0)
	model: AutoEncoder = AutoEncoder.from_pretrained(model_name, config=config)

	checkpoint = "checkpoint/65586150dd2ce3eba3172ea837b748286e277200/autoencoder_00_34253.pth"
	checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
	encoder_checkpoint = {
		k: v
		for k, v in checkpoint["model"].items()
		if k.startswith("encoder.")
	}

	checkpoint = "summarizer_00.pth"
	checkpoint = torch.load(checkpoint, map_location=device, weights_only=True)
	decoder_checkpoint = checkpoint["model"]

	checkpoint = encoder_checkpoint | decoder_checkpoint
	model.load_state_dict(checkpoint)
	model.eval()
	model.to(device)

	dataset = datasets.load_from_disk("embeddings/cnndm")
	dataset = dataset["test"]
	dataset = dataset.select_columns(
		["article", "highlights", "highlights_token", "article_embeds"]
		)  # TODO: highlights_tokens
	dataset = SummarizerDataset(
		dataset=dataset,
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
		num_workers=4,
	)

	rouge_values = []
	bar = tqdm(dataloader)
	for batch in bar:
		batch: SummarizerBatch = batch.to(device)

		outputs = model.generate(
			attention_mask=batch.pad_mask,
			encoder_outputs=Encoder.Output(
				last_hidden_state=batch.input_embeds,
			),
			use_cache=False,
			do_sample=False,
			num_beams=4,
		)

		outputs = tokenizer.batch_decode(
			outputs,
			skip_special_tokens=True,
		)

		results = rouge.compute(
			predictions=outputs,
			references=batch.labels_str,
			rouge_types=["rouge1"],  # , "rouge2", "rougeL", "rougeLsum"],
			use_aggregator=False,
		)

		rouge_values += results["rouge1"]
		rouge_mean = sum(rouge_values) / len(rouge_values)
		bar.set_postfix(r=rouge_mean)
	bar.close()

if __name__ == "__main__":
	main()
