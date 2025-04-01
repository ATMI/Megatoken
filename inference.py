from pathlib import Path

import numpy as np
import scipy as sp
import shap
import torch
from networkx.linalg.attrmatrix import attr_matrix
from torch import Tensor
from transformers import AutoTokenizer, T5Tokenizer

import prepare
from config import Config


def inference(
        model,
        tokenizer: T5Tokenizer,
        text,
):
    max_length = 128
    T = 1.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    tokens = tokenizer.__call__(
        text,
        padding=False,
        truncation=True,
        max_length=Config.max_length,
        return_attention_mask=False,
        return_tensors="pt",
    )["input_ids"].to(device)

    # tokens = torch.tensor([[3, 8691, 16, 8, 7281, 15, 12572, 2282, 6, 48, 706, 13442, 19, 1134, 1633, 12, 691, 91, 5, 3057, 19, 2595, 6, 128, 5436, 87, 391, 29, 279, 6, 68, 20, 89, 95, 12745, 3605, 78, 25, 470, 43, 12, 1190, 10410, 55, 20289, 7, 33, 15708, 713, 6, 68, 258, 541, 6, 2342, 207, 6750, 21, 3, 9, 7162, 594, 16, 7615, 19, 29, 31, 17, 24, 514, 5, 4886, 6, 3, 99, 25, 31, 60, 479, 21, 3, 9, 207, 97, 6, 27, 1568, 6450, 48, 286, 91, 5, 1]])

    model.eval()
    with torch.no_grad():
        memory = model.encode(
            tokens=tokens,
            pad_mask=torch.ones_like(tokens, dtype=torch.bool).to(device),
            attn_mask=None
        )

    print("Volume:", memory.volume)

    out = tokens[0][:Config.decoder_visibility+1].unsqueeze(0)

    while True:
        seq_length = out.size(1)
        decoder_mask = torch.full((seq_length, seq_length), -torch.inf)
        for i in range(seq_length):
            decoder_mask[i:i + Config.decoder_visibility + 1, i] = 0
        decoder_mask[:, 0] = 0


        with torch.no_grad():
            logits = model.decode(
                memory=memory,
                tokens=out,
                pad_mask=torch.ones_like(out, dtype=torch.bool).to(device),
                attn_mask=decoder_mask,
            ).squeeze(0)

        next_tok = logits[-1].argmax(dim=-1)
        # logits /= T
        # next_tok = logits.softmax(dim=-1)
        # next_tok = torch.multinomial(next_tok, num_samples=1)[-1]


        if next_tok.item() == tokenizer.eos_token_id:
            print("EOS!")
            break

        next_tok = torch.tensor([[next_tok]])
        out = torch.cat((out, next_tok), dim=1).to(device)

        if out.size(-1) == max_length:
            break

    output = tokenizer.decode(out.squeeze())
    print("Predicted:\n", output, sep="")


def load_ckpt(model, ckpt_path: Path | str):
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))

    model.load_state_dict(checkpoint["model"])
    print("Loaded!")
    return model


def viz(model, tokenizer, text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    input_ids = tokenizer(
        text,
        padding=False,
        truncation=True,
        max_length=Config.max_length,
        return_attention_mask=False,
        return_tensors="pt",
    )["input_ids"].to(device)

    model.eval()
    with torch.no_grad():
        memory = model.encode(
            tokens=input_ids,
            pad_mask=torch.ones_like(input_ids, dtype=torch.bool).to(device),
            attn_mask=None
        )

    attention_maps, gate_values = memory.attn_scores, memory.gates

    removed_token_indices = torch.tensor([], device=device)

    for layer_idx, (layer_attention, layer_gates) in enumerate(zip(attention_maps, gate_values)):
        print(f"Layer {layer_idx}:")

        # Find eliminated tokens at this layer!
        eliminated_indices = torch.where(layer_gates == -torch.inf)[0]
        mm = ~torch.isin(eliminated_indices, removed_token_indices)
        eliminated_indices = eliminated_indices[mm]
        if eliminated_indices.numel() == 0:
            print("No compression in this layer!\n")
            continue

        # Track eliminated tokens (to display shortened sequence)

        removed_token_indices = torch.cat((removed_token_indices, eliminated_indices))

        # Mean along all Attention Heads
        attention_matrix = layer_attention.mean(dim=0)

        # Find to where eliminated tokens were merged
        eliminated_attention = attention_matrix[eliminated_indices]
        topk_values, topk_indices = torch.topk(eliminated_attention, k=4, dim=1)

        # Convert indices to tokens
        eliminated_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][eliminated_indices])

        # Print eliminated tokens
        for token_idx, token in enumerate(eliminated_tokens):
            print(f"- Eliminated token: '{token}', (position: {eliminated_indices[token_idx]})")

            # Go through each parent token
            for rank, (value, parent_idx) in enumerate(
                    zip(topk_values[token_idx], topk_indices[token_idx]),
                    start=1
            ):
                parent_token = tokenizer.convert_ids_to_tokens(input_ids[0][parent_idx].item())
                print(f"\t\t{rank}. Merged into '{parent_token}' (score: {value.item():.4f}), (position: {parent_idx})")

        # Show compressed sequence
        seq_len = input_ids.size(1)
        active_mask = ~torch.isin(torch.arange(seq_len), removed_token_indices)
        compressed_sequence = tokenizer.decode(input_ids[0][active_mask])

        print(f"\nCompressed sequence after layer {layer_idx}:")
        print(input_ids[0][active_mask].shape)
        print(compressed_sequence)
        print("\n" + "-" * 50 + "\n")


    print(memory.gate_mask)
    print(memory.volume)

    # Final compressed output
    final_sequence = tokenizer.decode(input_ids[0][active_mask])

    print("\n" + "=" * 50 + "\n")
    print("Final compressed text:")
    print(final_sequence)

    print("\nInitial Sequence:")
    print(tokenizer.decode(input_ids[0]))


def main():
    tokenizer = AutoTokenizer.from_pretrained(Config.model)

    # text = "Good beer selection. Understaffed for a light Monday night crowd, it wasn't her fault she was the only server. But it took about an hour to get our sandwiches. Mine was one of the best reubens I've ever had."
    text = "Very disappointed in the customer service. We ordered Reuben's and wanted coleslaw instead of kraut. They charged us $3.00 for the coleslaw. We will not be back . The iced tea is also terrible tasting."
    # text = "Very nice restaurant! Will be back. Recommend trying Hawaiian pizza"
    # text = "Great breakfast, good price. You might have to stand outside in line though, so I don't really recommend winter time to go. lol. Very friendly service, interesting coffee mugs. They have great deserts and such also. Bring your cash though as they dont' take cards."
    # text = "Talk about overpriced. $18 for a fairly basic pasta with some obviously frozen chicken chopped up over it. The latter was terrible, thin and flabby and rather unappealing. The pasta itself was ok, as was the sauce. The desserts are pretty good. But honestly, that is a $10 dish whose price has been inflated."
    print("Initial text:", text, sep="\n")

    model = prepare.model()
    model = load_ckpt(model, "checkpoint/plak.pth")

    viz(model, tokenizer, [text])
    # inference(model, tokenizer, text)



if __name__ == '__main__':
    main()