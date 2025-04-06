from pathlib import Path

import torch
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
    T = 0.2

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
    pad_mask = torch.ones_like(tokens, dtype=torch.bool).to(device)


    # tokens = torch.tensor([[2940,594,47,269,11,8,313,47,1287,5,27,141,3,9,422,682,28,82,8179,11,79,237,2348,140,12,3,9,313,2478,12,143,417,27,530,132,7794,5,1333,7242,8833,5,305,4811,55,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    # print(tokenizer.decode(tokens[0]))
    # pad_mask = torch.zeros_like(tokens, dtype=torch.bool)
    # pad_mask[:, :45] = 1
    # tokens = tokens[:, :45]
    # pad_mask = pad_mask[:, :45]

    model.eval()
    with torch.no_grad():
        memory = model.encode(
            tokens=tokens,
            pad_mask=pad_mask,
            attn_mask=None
        )

    print("Volume:", memory.volume)

    out = tokens[0][:Config.decoder_visibility].unsqueeze(0)
    print(tokenizer.decode(out[0]))

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
        # print(input_ids[0][active_mask].shape)
        print(compressed_sequence)
        print("\n" + "-" * 50 + "\n")


    print(memory.gate_mask)
    print(input_ids.shape)
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
    # text = """THE WORST DENTAL EXPERIENCE OF MY LIFE.  THEY ARE BUTCHERS!  My husband and I went to Emmert Dental in Bethel Park for a routine cleaning and check up.  I had one tooth that I had a recent root canal on and a few small cavities that needed filled.  After the so called dentist - Dr Carnavali - did my exam, I needed 3 additional root canals and an extraction! An extraction on the tooth that just had a root canal. I was told that part of the tooth had broken off and that a crown was no longer possible.  I didn't know any better, so I agreed to schedule the extraction for a few weeks later. The \"dentist\" wanted to extract it that day, but I had plans for the weekend and was not about to be in pain. So, 2 weeks later, I went in for the extraction.  After they gave me novocaine they get me with \"you need a bone graft also in that area since we are extracting that tooth\". I have never had any issues with bone loss with previous extractions so I didn't feel it was necessary. I had 3 employees surrounding me, pressuring me for this bone graft. So, I reluctantly agreed. I had to pay $259.00 for this \"bone graft\". So, I signed everything. I wasn't even given enough time to read the document, so I should have known and got up and left right then and there. The \"dentist\" started the extraction. I have never been so uncomfortable in my life. He pulled on my tooth, used a metal instrument to pound on my tooth and was pulling so hard that my head was coming up and slamming back down against the head rest.  Finally, after a good 15 minutes of this torture, he ripped the tooth out of the side of my jaw. I was literally ganging from all the blood running down the back of my throat. He then stuck some pink powder up in the extraction area. I'm assuming this was the bone graft.  I ended up with many stitches. I was given an antibiotic and instructed to schedule an appointment for the following Saturday so they could start the bridge. Then I was hit with a bill of $569.00! My insurance covers 100% of extractions.  I was in so much pain for 2 weeks following the extraction. Pain severe enough that I had to call off work and I couldn't sleep at night.  I refused to go back to Emmert Dental.  And after a visit to a very reputable dentist, an endodontist and an oral surgeon, I have to have oral surgery to try and repair the damage Emmert Dental has done to my mouth.  They caused a very large deformity in my jaw and I am at risk to lose additional teeth.  Save yourself a lot of time, money and pain.  Emmert dental only cares about the money, will over charge you and leave you less than happy with the dental work.  They are butchers and should not even call themselves dentist. And if you have an issue with their billing practices, don't expect to that resolved easily.  I have called numerous times and I am always told that the manager and or billing department is not in.  I have contacted the BBB and a lawyer.  I expect and want a full refund.  No one should have to go thru what I am currently going thru.  They do unnecessary extractions so they can do a more expensive procedure such as a bridge or an implant.  Now I have to go thru oral surgery, more pain and suffering and more time missed from work because of Emmert Dental.  And the root canals they said I needed, my current dentist and my endodontist both agreed that I do not need tooth canals.  In fact, the teeth are perfectly healthy.  They are crooks and butchers."""
    print("Initial text:", text, sep="\n")

    model = prepare.model()
    model = load_ckpt(model, "checkpoint/32499.pth")

    # viz(model, tokenizer, [text])
    inference(model, tokenizer, text)



if __name__ == '__main__':
    main()