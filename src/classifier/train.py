import torch
from torch import nn
from torch import optim
from torch.utils import data
from tqdm import tqdm

from .batch import ClassifierBatch
from .dataset import ClassifierDataset
from .encoder import Encoder
from .model import Classifier

from ..util import metric
from ..util.metric import RollingMean
from ..util.prepare import prepare_random, prepare_device


def main():
	prepare_random()
	device = prepare_device()

	encoder = Encoder(
		checkpoint="autoencoder_00.pth",
		device=device,
	)

	dataset = ClassifierDataset(
		name="stanfordnlp/imdb",
		version=None,
		split="train",
		model_name=encoder.name,
		text_column="text",
		label_column="label",
	)

	dataloader = data.DataLoader(
		dataset=dataset,
		batch_size=32,
		shuffle=True,
		collate_fn=ClassifierBatch.collate_fn(encoder.pad_token),
	)

	classifier = Classifier(num_classes=2).train().to(device)
	optimizer = optim.Adam(
		params=classifier.parameters(),
		lr=1e-3,
	)
	scheduler = optim.lr_scheduler.StepLR(
		optimizer=optimizer,
		step_size=1,
		gamma=0.5,
	)

	criterion = nn.CrossEntropyLoss()
	rolling = RollingMean(50)

	epoch_num = 5
	for epoch in range(epoch_num):
		bar = tqdm(
			dataloader,
			leave=True,
			desc=f"Epoch {epoch + 1}/{epoch_num}"
		)

		for batch in bar:
			batch: ClassifierBatch = batch.to(device)
			embeds, indices = encoder(
				input_ids=batch.input_ids,
				attention_mask=batch.pad_mask,
			)

			optimizer.zero_grad()
			logits = classifier.forward(embeds, indices)

			loss = criterion(logits, batch.labels)
			loss.backward()
			optimizer.step()

			conf = metric.confusion(logits, batch.labels)
			acc, pre, rec = rolling(conf.accuracy, conf.precision, conf.recall)
			bar.set_postfix(acc=acc, pre=pre, rec=rec)

		bar.close()
		torch.save(classifier.state_dict(), f"{epoch}_classifier.pth")
		scheduler.step()


if __name__ == "__main__":
	main()
