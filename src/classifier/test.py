import torch
from torch.utils import data
from tqdm import tqdm

from .encoder import Encoder
from .batch import ClassifierBatch
from .dataset import ClassifierDataset
from .model import Classifier

from ..util import metric
from ..util.metric import RollingMean
from ..util.prepare import prepare_random, prepare_device


def main():
	prepare_random()
	device = prepare_device()

	encoder = Encoder(
		checkpoint="autoencoder_00_01901.pth",
		device=device,
	)

	dataset = ClassifierDataset(
		name="stanfordnlp/imdb",
		version=None,
		split="test",
		model_name=encoder.name,
		text_column="text",
		label_column="label",
	)

	dataloader = data.DataLoader(
		dataset=dataset,
		batch_size=64,
		shuffle=True,
		collate_fn=ClassifierBatch.collate_fn(encoder.pad_token),
	)

	classifier = Classifier(num_classes=2)
	classifier_init = "3_classifier.pth"
	classifier_init = torch.load(classifier_init, map_location="cpu", weights_only=True)
	classifier.load_state_dict(classifier_init)
	classifier = classifier.eval().to(device)

	rolling = RollingMean(-1)
	bar = tqdm(dataloader, leave=True)
	torch.set_grad_enabled(False)

	for batch in bar:
		batch: ClassifierBatch = batch.to(device)
		embeds, indices = encoder(
			input_ids=batch.input_ids,
			attention_mask=batch.pad_mask,
		)

		logits = classifier.forward(embeds, indices)
		conf = metric.confusion(logits, batch.labels)

		acc, pre, rec = rolling(conf.accuracy, conf.precision, conf.recall)
		bar.set_postfix(acc=acc, pre=pre, rec=rec)

	bar.close()


if __name__ == "__main__":
	main()
