from pathlib import Path

import numpy as np
import torch
from networkx.algorithms.centrality import trophic_levels
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import h5py

import prepare




def load_ckpt(model, ckpt_path: Path | str):
	checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))

	model.load_state_dict(checkpoint["model"])
	print("Loaded!")
	return model



def pack_sequence(embeds: Tensor, gate_mask, pad_mask):
	mask = (gate_mask.exp() * pad_mask).bool()

	lengths = mask.sum(dim=1).numpy().tolist()

	flat_mask = mask.flatten()


	embed_size = embeds.size(-1)
	flat_emb = embeds.view(-1, embed_size)
	filtered_embeds = flat_emb[flat_mask]

	cleaned_seq = torch.split(filtered_embeds, lengths)
	# out = pad_sequence(list(cleaned_seq), batch_first=True)
	print(mask.sum(dim=1).numpy().mean())
	return list(cleaned_seq)




def run():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	CoT = prepare.model()
	CoT = load_ckpt(CoT, "checkpoint/1.pth")
	CoT = CoT.to(device)
	CoT = CoT.eval()

	train_loader, _ = prepare.dataloaders()
	step_num = len(train_loader)
	bar = tqdm(total=step_num)


	with h5py.File("embeds.h5", "w") as f:
		tensor_group = f.create_group("embeds")

		for step, batch in enumerate(train_loader):
			batch = batch.to(device)
			x, y = batch.inputs, batch.sentiment
			pad_mask = batch.pad_mask

			with torch.no_grad():
				memory = CoT.encode(x, pad_mask, None)

			packed_seq = pack_sequence(memory.embeds, memory.gate_mask, memory.pad_mask)

			for i, tns in enumerate(packed_seq):
				tensor_group.create_dataset(f"embeds_{step}_{i}", data=tns.detach().numpy())

			if step % 5 == 0 and step != 0:
				break

			bar.update(1)



def check_tensors():
	tensors = {}
	known_steps = set()
	with h5py.File("embeds.h5", "r") as f:
		tensor_group = f["embeds"]
		for key in tensor_group.keys():
			step_num = int(key.split("_")[-2])

			if step_num in known_steps:
				continue
			pack = []
			for i in range(20):
				n_key = "embeds_{}_{}".format(step_num, i)

				numpy_arr = tensor_group[n_key][:]
				tns = torch.from_numpy(numpy_arr)
				pack.append(tns)

			out = pad_sequence(pack, batch_first=True)
			tensors[step_num] = out

	print("Loaded tensors!")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	CoT = prepare.model()
	CoT = load_ckpt(CoT, "checkpoint/1.pth")
	CoT = CoT.to(device)
	CoT = CoT.eval()

	train_loader, _ = prepare.dataloaders()
	step_num = len(train_loader)
	bar = tqdm(total=step_num)

	for step, batch in enumerate(train_loader):
		batch = batch.to(device)
		x, y = batch.inputs, batch.sentiment
		pad_mask = batch.pad_mask

		with torch.no_grad():
			memory = CoT.encode(x, pad_mask, None)

		packed_seq = pack_sequence(memory.embeds, memory.gate_mask, memory.pad_mask)
		packed_seq = pad_sequence(packed_seq, batch_first=True)

		if not torch.isclose(packed_seq, tensors[step], atol=1e-6).all():
			print("Err")

		if step % 5 == 0 and step != 0:
			break

		bar.update(1)


if __name__ == "__main__":
	run()
	check_tensors()