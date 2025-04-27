import torch
from transformers import AutoTokenizer

from .model import AutoEncoder, AutoEncoderConfig
from ..util.prepare import prepare_random, prepare_device


def main():
	prepare_random()
	device = prepare_device()

	model_name = "google/flan-t5-small"
	config = AutoEncoderConfig.from_pretrained(model_name)
	model = AutoEncoder(config).to(device)
	tokenizer = AutoTokenizer.from_pretrained(model_name)

	checkpoint = torch.load("autoencoder_00_34253.pth", map_location=device, weights_only=True)
	checkpoint = checkpoint["model"]
	checkpoint = {k.removeprefix("t5."): v for k, v in checkpoint.items()}
	model.load_state_dict(checkpoint)

	inputs = [
		"""(CNN) -- A truck-bomb explosion has killed at least 13 people in the capital of Afghanistan's Logar province, a provincial spokesman said. The attack Saturday night killed four civilians and several militants, including members of the Pakistani Taliban and an affiliated group, said Din Mohammad Darwis, the spokesman. It happened in Pul e Alam, in central Afghanistan.""",
		"""(CNN) One person was killed after more than 100 cars piled up on Interstate 94 near Kalamazoo, Michigan, state police said Friday. About 20 more were injured, with 10 of those injuries "more serious" in nature, Michigan State Police Officer Shane Criger said. One of the tractor-trailers involved in the wreck was carrying fireworks that can be seen on video detonating.""",
	]
	prefix = [i[:5] for i in inputs]

	inputs = tokenizer(inputs, return_tensors="pt", padding=True)
	inputs = inputs.to(device)

	prefix = tokenizer(prefix, return_tensors="pt", return_attention_mask=False)["input_ids"]
	prefix = prefix.to(device)
	inputs["decoder_input_ids"] = prefix

	outputs = model.generate(**inputs)
	outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

	print(outputs)


if __name__ == "__main__":
	main()
