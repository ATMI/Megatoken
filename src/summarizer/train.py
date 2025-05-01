import torch
from torch import optim
from torch.nn import functional as fn
from torch.utils import data
from tqdm import tqdm

from .batch import SummarizerBatch
from .dataset import SummarizerDataset
from .log import SummarizerLog
from ..autoencoder.model import AutoEncoderConfig, AutoEncoder
from ..util.metric import accuracy
from ..util.prepare import prepare_random, prepare_device


def main():
	prepare_random()
	device = prepare_device()

	epoch_num = 10
	model_name = "google/flan-t5-small"

	checkpoint = "autoencoder_01.pth"
	checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)

	config = AutoEncoderConfig.from_pretrained(model_name, decoder_visibility=0)
	model: AutoEncoder = AutoEncoder.from_pretrained(model_name, config=config)
	model.load_state_dict(checkpoint["model"])
	model.to(device)
	model.train()

	dataset = SummarizerDataset(
		split="train",
		model_name=model_name,
		bos_token=config.pad_token_id,
	)
	dataloader = data.DataLoader(
		dataset=dataset,
		batch_size=24,
		shuffle=True,
		collate_fn=SummarizerBatch.collate_fn(
			pad_token=config.pad_token_id,
			ign_token=config.ign_token_id,
		),
	)

	params = []
	freeze = ["encoder.", "shared."]
	for name, param in model.named_parameters():
		if any(name.startswith(prefix) for prefix in freeze):
			param.requires_grad = False
		else:
			params.append(param)

	optimizer = optim.Adam(
		params=params,
		lr=1e-4,
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
				input_ids=batch.article_tokens,
				attention_mask=batch.article_padding,
				decoder_input_ids=batch.decoder_tokens,
			)

			loss = fn.cross_entropy(
				input=output.logits.flatten(0, 1),
				target=batch.summary_tokens.flatten(),
				ignore_index=config.ign_token_id,
			)

			loss.backward()
			optimizer.step()

			acc = accuracy(output.logits, batch.summary_tokens, config.ign_token_id)
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
