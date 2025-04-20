import torch
from torch.utils import data
from tqdm import tqdm

from .batch import Batch
from .config import Config
from .dataset import Dataset
from .model import Classifier

from ..util import prepare
from ..util import metric
from ..util.metric import RollingMean


def main():
	prepare.rnd(Config.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	dataset = Dataset("test")
	dataloader = data.DataLoader(
		dataset=dataset,
		batch_size=Config.batch_size,
		collate_fn=Batch.collate,
	)

	model = Classifier()
	model = model.to(device)

	init = "classifier.pth"
	init = torch.load(init, map_location=device, weights_only=True)
	model.load_state_dict(init)

	bar = tqdm(dataloader)
	rolling = RollingMean(Config.rolling_n)

	for batch in bar:
		batch = batch.to(device)
		logits = model.forward(batch.embeds, batch.indices)

		conf = metric.confusion(logits, batch.labels)
		a, p, r = rolling(conf.accuracy, conf.precision, conf.recall)

		bar.set_postfix(a=a, p=p, r=r)


if __name__ == "__main__":
	main()
