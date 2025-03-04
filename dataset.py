from datasets import load_dataset
from transformers import AutoTokenizer

NUM_PROC = 4


def tokenize(batch, tokenizer):
	tokenizer = AutoTokenizer.from_pretrained(tokenizer)

	text = tokenizer(
		batch,
		return_attention_mask=False,
		return_token_type_ids=False,
		truncation=True,
	)
	text = {
		"tokens": text["input_ids"]
	}

	return text


def yelp_dataset(
	tokenizer: str,
	num_proc: int = NUM_PROC,
):
	ds = load_dataset("Yelp/yelp_review_full")
	ds = ds.map(
		tokenize,
		fn_kwargs={
			"tokenizer": tokenizer,
		},
		input_columns=["text"],
		num_proc=num_proc,
		batched=True,
	)
	ds = ds.select_columns(["tokens", "label"])
	return ds
