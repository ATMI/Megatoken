import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import prepare
from model import Model
from transformers import AutoTokenizer
from config import Config
import shap
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained(Config.model)
model = prepare.model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
memory = None

start_sent = "This cafe is a hidden gem! The cozy atmosphere and excellent coffee make it the perfect spot to relax and unwind. The staff are friendly and attentive, always ensuring that your cup is full."
tokenized = tokenizer.encode(start_sent)[:-1]
memory_ids = torch.tensor(tokenized).unsqueeze(0).to(device)
input_ids = memory_ids[:,:5]

checkpoint = torch.load("checkpoint.pth", map_location=device, weights_only=True)
model.eval()
model.load_state_dict(checkpoint["model"])
model = model.to(device)
correct_token_id = 5697


def shap_predict(masked_indices):
	"""
    Runs inference with a modified gate mask where specific tokens are masked.
    `masked_indices` is a binary mask of shape (seq_len,) indicating which tokens to mask.
    """
	masked_indices = torch.tensor(masked_indices, dtype=torch.bool, device=device)
	#print("received masked_indices:", masked_indices)

	valid_indices = (memory.gate_mask == 0)
	full_mask = torch.zeros_like(memory.gate_mask, dtype=torch.bool, device=device)
	full_mask[valid_indices] = masked_indices


	modified_gate_mask = memory.gate_mask.clone()

	modified_gate_mask[full_mask] = -torch.inf

	result = model.decode(
		memory=Model.Memory(
			pad_mask=memory.pad_mask,
			gate_mask=modified_gate_mask,
			embeds=memory.embeds,
			volume=memory.volume,
		),
		tokens=input_ids,
		pad_mask=torch.ones((1, input_ids.shape[1]), dtype=torch.bool).to(device),
		attn_mask=None
	)

	logits = result[:, -1, :]
	probs = torch.softmax(logits, dim=-1)
	correct_token_prob = probs[:, correct_token_id]

	return correct_token_prob.cpu().detach().numpy()



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
	global model, memory, input_ids
	torch.set_grad_enabled(False)


	#test_loader, t_loader  = prepare.dataloaders()

	memory = model.encode(
		tokens=memory_ids,
		pad_mask=torch.ones((1,memory_ids.shape[1]), dtype = torch.bool).to(device),
		attn_mask=None,
	)
	print("original memory: ", memory.gate_mask)
	original_gate_mask = memory.gate_mask
	valid_indices = (original_gate_mask == 0).squeeze()
	num_valid_tokens = valid_indices.sum().item()

	data = torch.zeros((1, num_valid_tokens))
	data = data.cpu().detach().numpy()

	explainer = shap.KernelExplainer(shap_predict, data)

	shap_values = explainer.shap_values(torch.eye(valid_indices.sum()).numpy())



	shap_values_np = np.array(shap_values).squeeze()
	avg_shap_values = np.mean(np.abs(shap_values_np), axis=0)
	plt.figure(figsize=(12, 5))
	plt.bar(range(len(avg_shap_values)), avg_shap_values)
	plt.xlabel("Token Index (Valid Ones)")
	plt.ylabel("SHAP Value (Importance)")
	plt.title("Token Importance in Cross-Attention (Gate Mask Analysis)")
	plt.show()



def generate_sentence(num_steps):
	global model,input_ids,memory_ids
	print("initial sequence: ", input_ids)
	le = input_ids.shape[-1]
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
	print("final tokens: ", input_ids)
	print("next token is: ", input_ids[0][le], "decoded as: ", tokenizer.decode(input_ids[0][le], skip_special_tokens=False))

if __name__ == "__main__":
	main()
