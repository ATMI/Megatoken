import datasets
import torch
from torch import optim
from torch.nn import functional as fn
from torch.utils import data
from tqdm import tqdm

from .dataset import SummarizerDataset
from .batch import SummarizerBatch
from .log import SummarizerLog

from ..autoencoder.encoder import Encoder
from ..autoencoder.model import AutoEncoderConfig, AutoEncoder
from ..util.prepare import prepare_random, prepare_device
from ..util.metric import accuracy


def main():
	prepare_random()
	device = prepare_device()

	epoch_num = 1
	model_name = "google/flan-t5-small"
	config = AutoEncoderConfig.from_pretrained(model_name, decoder_visibility=0)

	model: AutoEncoder = AutoEncoder.from_pretrained(model_name, config=config)
	del model.encoder

	checkpoint = "checkpoint/65586150dd2ce3eba3172ea837b748286e277200/autoencoder_00_34253.pth"
	checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
	checkpoint = {
		k: v
		for k, v in checkpoint["model"].items()
		if not k.startswith("encoder.")
	}

	model.load_state_dict(checkpoint)
	model.train()
	model.to(device)

	params = []
	for name, param in model.named_parameters():
		if name.startswith("shared."):
			param.requires_grad = False
		else:
			params.append(param)

	dataset = datasets.load_from_disk("embeddings/cnndm1")
	dataset = dataset.select_columns(["article_embeds", "highlights_embeds"])
	dataset = SummarizerDataset(
		dataset=dataset,
		bos_token=config.pad_token_id,
	)
	dataloader = data.DataLoader(
		dataset=dataset,
		batch_size=4,
		collate_fn=SummarizerBatch.collate_fn(
			pad_token=config.pad_token_id,
			ign_token=config.ign_token_id,
		),
	)

	optimizer = optim.Adam(
		params=params,
		lr=3e-4,
	)
	scheduler = optim.lr_scheduler.StepLR(
		optimizer=optimizer,
		step_size=1,
		gamma=0.5,
	)

	log = SummarizerLog(
		file="summarizer.log",
		rolling_n=100,
	)

	for epoch in range(epoch_num):
		bar = tqdm(
			iterable=dataloader,
			desc=f"Epoch {epoch + 1}/{epoch_num}",
			leave=True,
		)

		for batch in bar:
			optimizer.zero_grad()

			batch: SummarizerBatch = batch.to(device)
			output = model.forward(
				attention_mask=batch.pad_mask,
				encoder_outputs=Encoder.Output(
					last_hidden_state=batch.input_embeds,
				),
				decoder_input_ids=batch.input_tokens,
			)

			loss = fn.cross_entropy(
				input=output.logits.flatten(0, 1),
				target=batch.labels.flatten(),
				ignore_index=config.ign_token_id,
			)

			loss.backward()
			optimizer.step()

			acc = accuracy(output.logits, batch.labels, config.ign_token_id)
			loss = loss.item()

			postfix = log(acc, loss)
			bar.set_postfix(**postfix)

		checkpoint = {
			"model": model.state_dict(),
			"optimizer": optimizer.state_dict(),
			"scheduler": scheduler.state_dict(),
		}
		torch.save(checkpoint, f"summarizer_{epoch:02d}.pth")

		scheduler.step()
		bar.close()

	log.close()


if __name__ == "__main__":
	main()
