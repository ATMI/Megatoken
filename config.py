class Config:
	dataset = "Yelp/yelp_review_full"
	tokenizer = "google-bert/bert-base-uncased"

	vocab_size = 30522
	max_length = 512
	ignore_token = -100
	mask_token = 103
	pad_token = 0

	seed = 42
	lr = 0.0001
	batch_size = 16
	rolling_n = 100

	t5 = "google/flan-t5-small"
	volume_threshold = 0.1
	decoder_visibility = 3
	temperature = 0.1
	bias = 5
