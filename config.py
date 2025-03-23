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
	batch_size = 32
	rolling_n = 100

	model_dim = 512
	head_num = 8
	fc_dim = 2048
	activation = "gelu"

	encoder_layers = 6
	temperature = 0.1
	bias = 0

	decoder_layers = 2
	sparsity = 0.15
