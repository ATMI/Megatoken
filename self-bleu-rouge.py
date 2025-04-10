import json
import os

import pandas as pd
import torch
import evaluate
from tqdm import tqdm
from transformers import AutoTokenizer
import Levenshtein

import prepare
from config import Config


def run(
		model,
		dataloader,
		tokenizer,
		device,
):
	def finish():
		log_file.close()
		bar.close()
		exit(0)

	print("Loading metrics from HF...")
	bleu_metric = evaluate.load("bleu")
	rouge_metric = evaluate.load("rouge")
	print("Metrics loaded!")

	bleu_scores = []
	rouge_scores = []
	levenshtein_scores = []
	os.makedirs("result", exist_ok=True)
	log_file = open("result/log.json", "w")

	bar = tqdm(total=len(dataloader))

	try:
		for step, batch in enumerate(dataloader):
			batch = batch.to(device)
			with torch.no_grad():
				result = model.forward(
					memory_tokens=batch.inputs,
					memory_eos_mask=batch.eos_mask,
					memory_pad_mask=batch.pad_mask,
					memory_attn_mask=None,

					input_tokens=batch.inputs,
					input_pad_mask=batch.pad_mask,
					input_attn_mask=batch.decoder_mask,
				)

			pred_ids = result.logits.argmax(-1)
			predictions = tokenizer.batch_decode(
				pred_ids.tolist(),
				skip_special_tokens=False,
				clean_up_tokenization_spaces=True
			)
			predictions = clean_text(predictions, tokenizer.eos_token)

			# Prepare references: Replace ignore token with Pad token from labels
			y = batch.labels
			y = torch.where(y == Config.ignore_token, Config.pad_token, y)
			refs = tokenizer.batch_decode(
				y,
				skip_special_tokens=False,
				clean_up_tokenization_spaces=True
			)
			refs = clean_text(refs, tokenizer.eos_token)

			bleu = bleu_metric.compute(predictions=predictions, references=refs)['bleu']
			rouge = rouge_metric.compute(predictions=predictions, references=refs)
			levenshtein = compute_levenstein(predictions, refs)

			bleu_scores.append(bleu)
			rouge_scores.append(rouge['rougeLsum'])
			levenshtein_scores.append(levenshtein)

			file_log = {
				"bleu": bleu,
				"rougeLsum": rouge['rougeLsum'],
				"levenshtein": levenshtein,
			}

			log_file.write(json.dumps(file_log) + "\n")
			log_file.flush()

			log = {
				"bleu": sum(bleu_scores) / len(bleu_scores),
				"rougeLsum": sum(rouge_scores) / len(rouge_scores),
				"levenstein": sum(levenshtein_scores) / len(levenshtein_scores),
			}
			bar.update(1)
			bar.set_postfix(log)
	except Exception:
		finish()

	save_results(bleu_scores, rouge_scores, levenshtein_scores)
	finish()




def compute_levenstein(predictions, refs):
	lev = []
	for p, r in zip(predictions, refs):
		score = Levenshtein.distance(p, r)
		if len(r) > 0:
			lev.append(score / len(r))

	return sum(lev) / len(lev)


def clean_text(texts, eos_token):
	return [sent.split(eos_token)[0] for sent in texts]


def save_results(bleu_scores, rouge_scores, levenshtein_scores):
	final_bleu = sum(bleu_scores) / len(bleu_scores)
	print(f"BLEU: {final_bleu}")

	final_levenshtein = sum(levenshtein_scores) / len(levenshtein_scores)
	print(f"Levenstein: {final_levenshtein}")

	final_rouge = sum(rouge_scores) / len(rouge_scores)
	print(f"ROUGE: {final_rouge}")

	data = [final_bleu, final_rouge]
	columns = ["bleu", "rouge"]

	results = pd.DataFrame(data, columns=columns)
	results.to_csv("result/scores.csv", index=False)


def main():
	torch.manual_seed(Config.seed)
	torch.cuda.manual_seed(Config.seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	tokenizer = AutoTokenizer.from_pretrained(Config.model)
	_, test_loader = prepare.dataloaders()

	model = prepare.model().to(device)
	checkpoint = torch.load("checkpoint/29249.pth", map_location=device, weights_only=True)
	model.load_state_dict(checkpoint["model"])
	model.eval()

	run(model=model, dataloader=test_loader, tokenizer=tokenizer, device=device)


if __name__ == '__main__':
	main()
