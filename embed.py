import torch
from tqdm import tqdm

import binfile
import prepare


def main():
	torch.set_grad_enabled(False)
	prepare.rnd()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	train_loader, _ = prepare.dataloaders()

	model = prepare.model()
	model = model.eval()
	model = model.to(device)

	init = "checkpoint/5bf16cb1b3d90095a76b251632a5c78f6530cd2a/32499.pth"
	init = torch.load(init, map_location=device, weights_only=True)
	model.load_state_dict(init["model"])

	writer = binfile.Writer("embeds")
	for batch in tqdm(train_loader):
		batch = batch.to(device)
		result = model.encode(
			tokens=batch.inputs,
			eos_mask=batch.eos_mask,
			pad_mask=batch.pad_mask,
			attn_mask=None,
		)

		mask = (result.gate_mask > -1) & result.pad_mask
		embeds = result.embeds[mask]
		mask = mask.sum(dim=1)

		embeds = embeds.cpu()
		mask = mask.cpu()

		head = 0
		for id, length in zip(batch.ids, mask):
			tail = head + length
			data = embeds[head:tail]
			head = tail
			writer.write(id, data)
	writer.close()


if __name__ == "__main__":
	main()
