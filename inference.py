import torch
from transformers import AutoTokenizer, T5Tokenizer

import prepare
from config import Config


def inference(
		model: torch.nn.Module,
		tokenizer: T5Tokenizer,
		text: str,
		max_length: int = 512,
):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)

	tokens = tokenizer(
		text,
		padding=False,
		truncation=True,
		max_length=Config.max_length,
		return_tensors="pt",
	)["input_ids"].to(device)
	pad_mask = torch.ones_like(tokens, dtype=torch.bool).to(device)
	encoder_eos_mask = torch.tensor([[0], [tokens.size(1) - 1]])

	print("Initial length:", tokens.size(1))

	model.eval()
	with torch.no_grad():
		memory = model.encode(
			tokens=tokens,
			pad_mask=pad_mask,
			attn_mask=None,
			eos_mask=encoder_eos_mask,
		)

	out_size = (memory.gate_mask.exp() != 0).sum().item()
	ratio = out_size / tokens.size(1)
	print(f"Compression rate: {ratio:.2f}")
	print("Output size:", out_size)

	out = tokens[0][:Config.decoder_visibility].unsqueeze(0)

	while True:
		seq_length = out.size(1)
		decoder_mask = torch.full((seq_length, seq_length), -torch.inf)
		for i in range(seq_length):
			decoder_mask[i:i + Config.decoder_visibility + 1, i] = 0
		decoder_mask[:, 0] = 0

		decoder_eos_mask = torch.tensor([[0], [out.size(1) - 1]])
		with torch.no_grad():
			logits = model.decode(
				memory=memory,
				tokens=out,
				pad_mask=torch.ones_like(out, dtype=torch.bool).to(device),
				attn_mask=decoder_mask,
				eos_mask=decoder_eos_mask,
			).squeeze(0)[-1]

		next_tok = logits.argmax(dim=-1)

		if next_tok.item() == tokenizer.eos_token_id:
			print("\nModel reached EOS token!")
			break

		next_tok = torch.tensor([[next_tok]])
		out = torch.cat((out, next_tok), dim=1).to(device)

		if out.size(-1) == max_length:
			break

	output = tokenizer.decode(out.squeeze())
	print("")
	print("Initial text:\n", text, sep="")
	print("Predicted:\n", output, sep="")


def main():
	tokenizer = AutoTokenizer.from_pretrained(Config.model)

	# text = "Good beer selection. Understaffed for a light Monday night crowd, it wasn't her fault she was the only server. But it took about an hour to get our sandwiches. Mine was one of the best reubens I've ever had."
	text = "Very disappointed in the customer service. We ordered Reuben's and wanted coleslaw instead of kraut. They charged us $3.00 for the coleslaw. We will not be back . The iced tea is also terrible tasting."
	# text = "Very nice restaurant! Will be back. Recommend trying Hawaiian pizza"
	# text = "Great breakfast, good price. You might have to stand outside in line though, so I don't really recommend winter time to go. lol. Very friendly service, interesting coffee mugs. They have great deserts and such also. Bring your cash though as they dont' take cards."
	# text = "Talk about overpriced. $18 for a fairly basic pasta with some obviously frozen chicken chopped up over it. The latter was terrible, thin and flabby and rather unappealing. The pasta itself was ok, as was the sauce. The desserts are pretty good. But honestly, that is a $10 dish whose price has been inflated."
	# text = """THE WORST DENTAL EXPERIENCE OF MY LIFE.  THEY ARE BUTCHERS!  My husband and I went to Emmert Dental in Bethel Park for a routine cleaning and check up.  I had one tooth that I had a recent root canal on and a few small cavities that needed filled.  After the so called dentist - Dr Carnavali - did my exam, I needed 3 additional root canals and an extraction! An extraction on the tooth that just had a root canal. I was told that part of the tooth had broken off and that a crown was no longer possible.  I didn't know any better, so I agreed to schedule the extraction for a few weeks later. The \"dentist\" wanted to extract it that day, but I had plans for the weekend and was not about to be in pain. So, 2 weeks later, I went in for the extraction.  After they gave me novocaine they get me with \"you need a bone graft also in that area since we are extracting that tooth\". I have never had any issues with bone loss with previous extractions so I didn't feel it was necessary. I had 3 employees surrounding me, pressuring me for this bone graft. So, I reluctantly agreed. I had to pay $259.00 for this \"bone graft\". So, I signed everything. I wasn't even given enough time to read the document, so I should have known and got up and left right then and there. The \"dentist\" started the extraction. I have never been so uncomfortable in my life. He pulled on my tooth, used a metal instrument to pound on my tooth and was pulling so hard that my head was coming up and slamming back down against the head rest.  Finally, after a good 15 minutes of this torture, he ripped the tooth out of the side of my jaw. I was literally ganging from all the blood running down the back of my throat. He then stuck some pink powder up in the extraction area. I'm assuming this was the bone graft.  I ended up with many stitches. I was given an antibiotic and instructed to schedule an appointment for the following Saturday so they could start the bridge. Then I was hit with a bill of $569.00! My insurance covers 100% of extractions.  I was in so much pain for 2 weeks following the extraction. Pain severe enough that I had to call off work and I couldn't sleep at night.  I refused to go back to Emmert Dental.  And after a visit to a very reputable dentist, an endodontist and an oral surgeon, I have to have oral surgery to try and repair the damage Emmert Dental has done to my mouth.  They caused a very large deformity in my jaw and I am at risk to lose additional teeth.  Save yourself a lot of time, money and pain.  Emmert dental only cares about the money, will over charge you and leave you less than happy with the dental work.  They are butchers and should not even call themselves dentist. And if you have an issue with their billing practices, don't expect to that resolved easily.  I have called numerous times and I am always told that the manager and or billing department is not in.  I have contacted the BBB and a lawyer.  I expect and want a full refund.  No one should have to go thru what I am currently going thru.  They do unnecessary extractions so they can do a more expensive procedure such as a bridge or an implant.  Now I have to go thru oral surgery, more pain and suffering and more time missed from work because of Emmert Dental.  And the root canals they said I needed, my current dentist and my endodontist both agreed that I do not need tooth canals.  In fact, the teeth are perfectly healthy.  They are crooks and butchers."""
	# text = "I got 'new' tires from them and within two weeks got a flat. I took my car to a local mechanic to see if i could get the hole patched, but they said the reason I had a flat was because the previous patch had blown - WAIT, WHAT? I just got the tire and never needed to have it patched? This was supposed to be a new tire. \nI took the tire over to Flynn's and they told me that someone punctured my tire, then tried to patch it. So there are resentful tire slashers? I find that very unlikely. After arguing with the guy and telling him that his logic was far fetched he said he'd give me a new tire \"this time\". \nI will never go back to Flynn's b/c of the way this guy treated me and the simple fact that they gave me a used tire!"

	checkpoint = torch.load("checkpoint/29249.pth", map_location=torch.device("cpu"))
	model = prepare.model()
	model.load_state_dict(checkpoint["model"])

	inference(model, tokenizer, text)


if __name__ == '__main__':
	main()
