import numpy as np
import torch
from transformers import AutoTokenizer

import networkx as nx
import matplotlib.pyplot as plt

from ..autoencoder.autoencoder import AutoEncoder
from ..autoencoder.config import Config


def main():
	# torch.set_grad_enabled(False)
	# sample = "Nick's has legit New York Style pizza. It's dirt cheap too!"
	#
	# init = "checkpoint/b054db9d3ff4941372c421d0286c7a405e91f036/1.pth"
	# init = torch.load(init, weights_only=True)
	#
	# model = Model(Config.model, Config.bias, Config.temperature)
	# model.eval()
	# model.load_state_dict(init["model"])
	#
	# tokenizer = AutoTokenizer.from_pretrained(Config.model)
	# tokens = tokenizer(
	# 	sample,
	# 	padding=False,
	# 	truncation=True,
	# 	max_length=Config.max_length,
	# 	return_tensors="pt",
	# )["input_ids"]
	#
	# eos_mask = torch.tensor([[0], [tokens.size(1) - 1]], dtype=torch.long)
	# pad_mask = torch.ones_like(tokens, dtype=torch.bool)
	#
	# memory = model.encode(
	# 	tokens=tokens,
	# 	eos_mask=eos_mask,
	# 	pad_mask=pad_mask,
	# 	attn_mask=None,
	# 	attn_scores=True,
	# )
	#
	# pad_mask = memory.pad_mask.detach().cpu()
	# gate_masks = memory.gate_masks.detach().cpu()
	# attn_scores = memory.attn_scores.detach().cpu()
	# tokens = tokens.detach().cpu()

	# with open("dump.npy", "wb") as f:
	# 	np.save(f, pad_mask)
	# 	np.save(f, gate_masks)
	# 	np.save(f, attn_scores)
	# 	np.save(f, tokens)
	# exit(0)
	#
	with open("dump.npy", "rb") as f:
		pad_mask = np.load(f)
		gate_masks = np.load(f)
		attn_scores = np.load(f)
		tokens = np.load(f)

	pad_mask = torch.from_numpy(pad_mask)
	gate_masks = torch.from_numpy(gate_masks)
	attn_scores = torch.from_numpy(attn_scores)
	tokens = torch.from_numpy(tokens)
	#
	pad_mask = pad_mask.squeeze(0)
	gate_masks = gate_masks.squeeze(1)
	attn_scores = attn_scores.mean(dim=1)  # [:, 5] #
	tokens = tokens.squeeze(0)

	g = nx.Graph()
	token_num = tokens.size(0)
	layer_num = 8
	thresh = 0.1

	gate_masks = (gate_masks > -1) & pad_mask
	for layer in range(layer_num + 1):
		for token in range(token_num):
			if layer > 0 and not gate_masks[(layer - 1) // 2][token]:
				continue
			g.add_node((layer, token), pos=(layer, -token))

	plt.figure(figsize=(20, 20))
	pos = nx.get_node_attributes(g, "pos")
	nx.draw_networkx_nodes(g, pos, nodelist=pos, node_size=500)

	for layer in range(layer_num - 1, -1, -1):
		scores = attn_scores[layer]
		mask = gate_masks[layer // 2].unsqueeze(1)
		dsts, srcs = torch.where((scores > thresh) & mask)

		for src, dst in zip(srcs, dsts):
			src, dst = src.item(), dst.item()
			score = scores[dst, src].item()
			g.add_edge((layer, src), (layer + 1, dst), score=score)

	edges = g.edges()
	scores = [g[u][v]["score"] for u, v in edges]
	colors = [plt.cm.Blues(w * 0.95 + 0.05) for w in scores]
	nx.draw_networkx_edges(
		g, pos,
		edgelist=edges,
		edge_color=colors,
		width=3,
		alpha=1,
	)

	plt.axis("off")
	plt.show()


if __name__ == "__main__":
	main()
