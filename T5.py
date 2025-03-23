from transformers import T5ForConditionalGeneration,AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import torch





# Set seed for reproducibility
def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




class CustomFLANT5WithGates(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, **kwargs):

        inputs_embeds = self.encoder.embed_tokens(input_ids)
        hidden_states = self.encoder.dropout(inputs_embeds)

        if attention_mask is not None:
            causal_mask = attention_mask[:, None, None, :]
            causal_mask = causal_mask.to(dtype=inputs_embeds.dtype)
            causal_mask = (1.0 - causal_mask) * torch.finfo(inputs_embeds.dtype).min
        else:
            causal_mask = None

        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

        batch_size, seq_length = input_shape
        cache_position = None
        past_key_values = None
        position_bias=None
        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0


        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, device=inputs_embeds.device
            )


        for i, encoder_layer in enumerate(self.encoder.block):

            print(f"Processing encoder layer {i + 1}")

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                cache_position=cache_position,
                position_bias=position_bias
            )
            layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            hidden_states, next_decoder_cache = layer_outputs[:2]
            position_bias = layer_outputs[2]
        hidden_states = self.encoder.final_layer_norm(hidden_states)
        hidden_states = (self.encoder.dropout(hidden_states))

        # wrapping
        encoder_outputs = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,  # This is what was originally returned by the encoder
            past_key_values=None,
            hidden_states=None,
            attentions=None
        )

        print("processed:",encoder_outputs)
        print("-------------")
        print("original:", self.encoder(input_ids=input_ids, attention_mask=attention_mask))
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs= encoder_outputs,
            **kwargs
        )


model_name = "google/flan-t5-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

custom_model = CustomFLANT5WithGates.from_pretrained(model_name).to(device)
original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# Fix the seed
set_seed(42)

tokenizer = AutoTokenizer.from_pretrained(model_name)

input_text = ["This is a test.", "How are you?"]  # Sample inputs
inputs = tokenizer(input_text, return_tensors="pt", padding=True)

input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)

decoder_input_ids = torch.full(
    (input_ids.shape[0], 1), tokenizer.pad_token_id, dtype=torch.long
).to(device)

custom_output = custom_model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
original_output = original_model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
print("---final---")
print("Custom Model Output:", custom_output.logits)
print("Original Model Output:", original_output.logits)
