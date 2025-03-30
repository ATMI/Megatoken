import torch
from tqdm import tqdm

import prepare
from model import Model
from transformers import AutoTokenizer
from config import Config


tokenizer = AutoTokenizer.from_pretrained(Config.model)

def show_result(batch,logits_batch):
	for i in range(logits_batch.shape[0]):
		inputs = batch.inputs[i].unsqueeze(0).expand(batch.inputs[i].shape[0],batch.inputs[i].shape[0])
		labels = batch.labels[i]
		decoder_mask = batch.decoder_mask[i]
		logits = logits_batch[i] # S,V
		masked_inputs = torch.where(decoder_mask == 0, inputs, torch.tensor(-100))
		print(logits.shape)
		logits = torch.argmax(logits, dim=-1)

		decoded_texts = [
			f"{tokenizer.decode(row[row != -100].tolist(), skip_special_tokens=True)}"
			f" [{tokenizer.decode([labels[j].item()], skip_special_tokens=True)}]"
			f"(predicted: [{tokenizer.decode([logits[j].item()], skip_special_tokens=True)}])"
			if labels[j] != -100 else ""
			for j, row in enumerate(masked_inputs)
		]

		print(decoded_texts)
		break



def main():
	torch.set_grad_enabled(False)
	num_steps = 200

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	#test_loader, t_loader  = prepare.dataloaders()

	checkpoint = torch.load("checkpoint.pth", map_location=device, weights_only=True)

	model = prepare.model()
	model.eval()
	model.load_state_dict(checkpoint["model"])
	model = model.to(device)
	start_sent = "This cafe is a hidden gem! The cozy atmosphere and excellent coffee make it the perfect spot to relax and unwind. The staff are friendly and attentive, always ensuring that your cup is full."
	tokenized = tokenizer.encode(start_sent)[:-1]
	memory_ids = torch.tensor(tokenized).unsqueeze(0).to(device)

	input_ids = memory_ids[:,:4]
	for i in range(num_steps):
		result = model.forward(
			memory_tokens = memory_ids,
			memory_pad_mask = torch.ones((1,memory_ids.shape[1]), dtype = torch.bool).to(device),
			memory_attn_mask = None,

			input_tokens = input_ids,
			input_pad_mask = torch.ones((1,input_ids.shape[1]), dtype = torch.bool).to(device),
			input_attn_mask= None

		)

		logits = result.logits[:,-1,:]
		prediction = torch.argmax(logits,dim=-1).unsqueeze(0)
		input_ids = torch.cat((input_ids,prediction),dim=-1)

	print(tokenizer.decode(input_ids.squeeze(0).tolist(), skip_special_tokens=False))



if __name__ == "__main__":
	main()
