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
import warnings
from sklearn.exceptions import ConvergenceWarning
import seaborn as sns

warnings.filterwarnings("ignore", category=ConvergenceWarning)
torch.set_grad_enabled(False)
INIT_TOKENS = 1

def plot_seaborn(sentence, influence, key_words, skip_tokens=0, save_path=None):
    assert influence.shape == (len(key_words), len(sentence) - skip_tokens), \
        f"Influence shape must be ({len(key_words)}, {len(sentence) - skip_tokens})"

    n_total = len(sentence)
    k = len(key_words)

    full_influence = np.full((k, n_total), np.nan)
    full_influence[:, skip_tokens:] = influence

    fig_width = max(8, 0.7 * n_total)
    fig_height = max(2.5, 0.6 * k)

    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.heatmap(
        full_influence,
        annot=[[f"{val:.2f}" if not np.isnan(val) else "" for val in row] for row in full_influence],
        xticklabels=sentence,
        yticklabels=key_words,
        cmap="RdBu",
        center=0,
        linewidths=0.5,
        cbar_kws={'label': 'Influence Score'},
        fmt=""
    )

    for x in range(skip_tokens):
        ax.add_patch(plt.Rectangle((x, -0.5), 1, k, fill=True, color='lightgray', alpha=0.3, lw=0))

    plt.title("Influence of Key Words on Sentence Tokens")
    plt.xlabel("Decoded Sentence")
    plt.ylabel("Key Words")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()



def plot(results):
    shap_values, token = results
    shap_values_np = np.array(shap_values).squeeze()

    #reversed_shap_values = -shap_values_np  # No need for diagonal extraction

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(shap_values_np)), shap_values_np)
    plt.ylim(-1, 1)
    plt.xlabel("Token Index (Valid Ones)")
    plt.ylabel("SHAP Value (Reversed Importance)")
    plt.title(f"Reversed Token Importance in Cross-Attention (Gate Mask Analysis) for token '{token}'")
    plt.show()


class SHAPexplainer():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(Config.model)
        self.model = prepare.model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load("checkpoint.pth", map_location=self.device, weights_only=True)
        self.model.eval()
        self.model.load_state_dict(checkpoint["model"])
        self.model = self.model.to(self.device)

        # buffer
        self.correct_token_id = None
        self.memory = None
        self.input_ids = None

    def shap_eval(self, sentence, steps):
        tokenized = self.tokenizer.encode(sentence)[:-1]
        memory_ids = torch.tensor(tokenized).unsqueeze(0).to(self.device)
        self.memory = self.model.encode(
            tokens=memory_ids,
            pad_mask=torch.ones((1, memory_ids.shape[1]), dtype=torch.bool).to(self.device),
            attn_mask=None,
        )
        self.input_ids = memory_ids[:, :1]

        tokens = self.generate_tokens(steps, memory_ids)
        decoded_tokens = [self.tokenizer.decode([token], skip_special_tokens=False) for token in tokens[0]]
        print(tokens)
        print(decoded_tokens)

        values_array = []
        for i in range(INIT_TOKENS, steps+1):
            self.input_ids = tokens[:, :i]
            self.correct_token_id = tokens[:, i]
            data, total_inds = self.prepare_data()
            explainer = shap.KernelExplainer(self.shap_predict, data)

            shap_values = explainer.shap_values(
                np.zeros((1, total_inds)),
                nsamples=300
            )
            values_array.append([shap_values, self.tokenizer.decode(self.correct_token_id)])

        decoded_memory = [self.tokenizer.decode([token], skip_special_tokens=False) for token in memory_ids[self.memory.gate_mask==0]]


        return values_array, decoded_tokens, decoded_memory

    def generate_tokens(self, num_steps, memory_ids):
        tokens = self.input_ids.clone()
        for i in range(num_steps):
            result = self.model.forward(
                memory_tokens=memory_ids,
                memory_pad_mask=torch.ones((1, memory_ids.shape[1]), dtype=torch.bool).to(self.device),
                memory_attn_mask=None,

                input_tokens=tokens,
                input_pad_mask=torch.ones((1, tokens.shape[1]), dtype=torch.bool).to(self.device),
                input_attn_mask=None

            )

            logits = result.logits[:, -1, :]
            prediction = torch.argmax(logits, dim=-1).unsqueeze(0)
            tokens = torch.cat((tokens, prediction), dim=-1)

        return tokens

    def shap_predict(self, masked_indices):
        """
        Runs inference with modified gate masks. Handles cases where all tokens are masked.
        Returns baseline probability (1/vocab_size) when all tokens are masked, using tokenizer's vocab size.
        """
        masked_indices = torch.tensor(masked_indices, dtype=torch.bool, device=self.device)
        if masked_indices.ndim == 1:
            masked_indices = masked_indices.unsqueeze(0)

        batch_size = masked_indices.shape[0]
        valid_indices = (self.memory.gate_mask == 0)
        vocab_size = len(self.tokenizer.vocab)
        correct_token_probs = []

        for i in range(batch_size):
            current_mask = masked_indices[i]
            if current_mask.all():
                uniform_prob = 1.0 / vocab_size
                correct_token_probs.append(torch.tensor([[uniform_prob]], device=self.device))
                continue
            full_mask = torch.zeros_like(self.memory.gate_mask, dtype=torch.bool, device=self.device)
            full_mask[valid_indices] = current_mask

            modified_gate_mask = self.memory.gate_mask.clone()
            modified_gate_mask[full_mask] = -torch.inf

            result = self.model.decode(
                memory=Model.Memory(
                    pad_mask=self.memory.pad_mask,
                    gate_mask=modified_gate_mask,
                    embeds=self.memory.embeds,
                    volume=self.memory.volume,
                ),
                tokens=self.input_ids,
                pad_mask=torch.ones((1, self.input_ids.shape[1]), dtype=torch.bool).to(self.device),
                attn_mask=None
            )

            logits = result[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            correct_token_prob = probs[:, self.correct_token_id]
            correct_token_probs.append(correct_token_prob)


        return torch.stack(correct_token_probs).cpu().detach().numpy()

    def prepare_data(self):
        original_gate_mask = self.memory.gate_mask
        valid_indices = (original_gate_mask == 0).squeeze()
        total_inds = valid_indices.sum()
        num_valid_tokens = total_inds.item()
        data = torch.ones((1, num_valid_tokens))
        data = data.cpu().detach().numpy()
        return data, total_inds

if __name__ == "__main__":
    start_sent = "This cafe is a hidden gem! The cozy atmosphere and excellent coffee make it the perfect spot to relax and unwind. The staff are friendly and attentive, always ensuring that your cup is full."
    shaper = SHAPexplainer()
    steps = 25
    results, decoded_sent, token_ids = shaper.shap_eval(start_sent, steps)
    raw_results = [results[i][0] for i in range(len(results))]
    accumulated_results = np.squeeze(np.concatenate(raw_results,axis = 2),axis = 0)

    plot_seaborn(decoded_sent, accumulated_results, token_ids, skip_tokens=INIT_TOKENS)

