import torch
from transformers import AutoTokenizer

import networkx as nx
import matplotlib.pyplot as plt

from ..autoencoder.model import Model
from ..autoencoder.config import Config


def main():
	torch.set_grad_enabled(False)
	# sample = "Visited this cozy Italian spot. Pizza & gelato were delicious! Service was good but a bit slow. Beautiful atmosphere with a classical furnace. Highly recommend for an authentic experience."
	sample = "We will not be back. The iced tea was terrible tasting. Disappointed in the customer service."

	init = "checkpoint/5bf16cb1b3d90095a76b251632a5c78f6530cd2a/29249.pth"
	init = torch.load(init, weights_only=True)

	model = Model(Config.model, Config.bias, Config.temperature)
	model.eval()
	model.load_state_dict(init["model"])

	tokenizer = AutoTokenizer.from_pretrained(Config.model)
	tokens = tokenizer(
		sample,
		padding=False,
		truncation=True,
		max_length=Config.max_length,
		return_tensors="pt",
	)["input_ids"]

	eos_mask = torch.tensor([[0], [tokens.size(1) - 1]], dtype=torch.long)
	pad_mask = torch.ones_like(tokens, dtype=torch.bool)

	memory = model.encode(
		tokens=tokens,
		eos_mask=eos_mask,
		pad_mask=pad_mask,
		attn_mask=None,
		attn_scores=True,
	)

	g = nx.Graph()
	token_num = tokens.size(1)
	thresh = 0.05

	# for layer in range(layer_num):
	for token in range(token_num):
		g.add_node((0, token), pos=(0, token))

	gate_masks = memory.gate_masks.squeeze(1)
	masks = (gate_masks > -1) & memory.pad_mask

	# attn_scores = memory.attn_scores[:, 5]
	attn_scores = memory.attn_scores.mean(dim=1)
	for layer, attn_scores in enumerate(attn_scores):
		if layer > 0:
			mask = masks[(layer - 1) // 2].unsqueeze(1)
			srcs, dsts = torch.where((attn_scores > thresh) & mask)
		else:
			srcs, dsts = torch.where(attn_scores > thresh)

		for src, dst in zip(srcs, dsts):
			src, dst = src.item(), dst.item()
			score = attn_scores[src, dst].item()

			src = (layer, src)
			dst = (layer + 1, dst)

			g.add_node(src, pos=src)
			g.add_node(dst, pos=dst)
			g.add_edge(src, dst, score=score)

	plt.figure(figsize=(20, 20))
	pos = nx.get_node_attributes(g, "pos")
	nx.draw_networkx_nodes(
		g, pos,
		nodelist=pos,
		node_size=500,
	)

	edges = g.edges()
	scores = [g[u][v]["score"] for u, v in edges]
	colors = [plt.cm.Blues(w) for w in scores]
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
