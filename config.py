from transformers import AutoTokenizer


class Config:
	dataset = "Yelp/yelp_review_full"
	model = "google/flan-t5-small"

	vocab_size: int = None
	max_length: int = None
	mask_token: int = None
	pad_token: int = None
	ignore_token = -100

	seed = 42
	lr = 0.0001
	batch_size = 20
	rolling_n = 10

	decoder_visibility = 5
	temperature = 0.1
	bias = 5


tokenizer = AutoTokenizer.from_pretrained(Config.model)
Config.vocab_size = tokenizer.vocab_size
Config.max_length = tokenizer.model_max_length
Config.pad_token = tokenizer.pad_token_id
Config.mask_token = tokenizer.mask_token_id
