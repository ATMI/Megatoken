import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoConfig

from .model import AutoEncoder, AutoEncoderConfig
from ..util.prepare import prepare_random, prepare_device


def main():
	prepare_random()
	device = prepare_device()

	model_name = "google/flan-t5-small"
	tokenizer = AutoTokenizer.from_pretrained(model_name)

	config = AutoEncoderConfig.from_pretrained(model_name)
	model = AutoEncoder.from_pretrained(model_name, config=config)
	model = model.to(device)

	checkpoint = torch.load("autoencoder_00_14000.pth", map_location=device, weights_only=True)
	model.load_state_dict(checkpoint["model"])

	inputs = [
		"""(CNN) -- A truck-bomb explosion has killed at least 13 people in the capital of Afghanistan's Logar province, a provincial spokesman said. The attack Saturday night killed four civilians and several militants, including members of the Pakistani Taliban and an affiliated group, said Din Mohammad Darwis, the spokesman. It happened in Pul e Alam, in central Afghanistan.""",
		"""(CNN) One person was killed after more than 100 cars piled up on Interstate 94 near Kalamazoo, Michigan, state police said Friday. About 20 more were injured, with 10 of those injuries "more serious" in nature, Michigan State Police Officer Shane Criger said. One of the tractor-trailers involved in the wreck was carrying fireworks that can be seen on video detonating.""",
	]

	inputs = tokenizer(inputs, return_tensors="pt", padding=True)
	inputs["use_cache"] = False
	inputs = inputs.to(device)

	outputs = model.generate(
		**inputs,
		max_length=128,
		eos_token_id=model.eos_token,
		do_sample=False,
		num_beams=1,
	)
	outputs = tokenizer.batch_decode(
		outputs,
		skip_special_tokens=True,
	)

	for output in outputs:
		print(output)


if __name__ == "__main__":
	main()
