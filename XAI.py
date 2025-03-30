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

torch.set_grad_enabled(False)


def plot(results):
    shap_values, token = results
    shap_values_np = np.array(shap_values).squeeze()
    individual_shap_values = np.diag(shap_values_np)
    reversed_shap_values = -individual_shap_values

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(reversed_shap_values)), reversed_shap_values)
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
        tokenized = self.tokenizer.encode(start_sent)[:-1]
        memory_ids = torch.tensor(tokenized).unsqueeze(0).to(self.device)
        self.memory = self.model.encode(
            tokens=memory_ids,
            pad_mask=torch.ones((1, memory_ids.shape[1]), dtype=torch.bool).to(self.device),
            attn_mask=None,
        )
        self.input_ids = memory_ids[:, :1]

        tokens = self.generate_tokens(steps, memory_ids)
        print(tokens)
        print(self.tokenizer.decode(tokens.squeeze(0).tolist(), skip_special_tokens=False))
        values_array = []
        for i in range(1, steps):
            self.input_ids = memory_ids[:, :i]
            self.correct_token_id = tokens[:, i]
            data, total_inds = self.prepare_data()
            explainer = shap.KernelExplainer(self.shap_predict, data)
            shap_values = explainer.shap_values(torch.eye(total_inds).numpy())
            values_array.append([shap_values, self.tokenizer.decode(self.correct_token_id, skip_special_tokens=False)])

        return values_array

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
        Runs inference with a modified gate mask where specific tokens are masked.
        `masked_indices` is a binary mask of shape (seq_len,) indicating which tokens to mask.
        """
        masked_indices = torch.tensor(masked_indices, dtype=torch.bool, device=self.device)
        # print("received masked_indices:", masked_indices)

        valid_indices = (self.memory.gate_mask == 0)
        full_mask = torch.zeros_like(self.memory.gate_mask, dtype=torch.bool, device=self.device)
        full_mask[valid_indices] = masked_indices

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

        return correct_token_prob.cpu().detach().numpy()

    def prepare_data(self):
        original_gate_mask = self.memory.gate_mask
        valid_indices = (original_gate_mask == 0).squeeze()
        total_inds = valid_indices.sum()
        num_valid_tokens = total_inds.item()
        data = torch.zeros((1, num_valid_tokens))
        data = data.cpu().detach().numpy()
        return data, total_inds


if __name__ == "__main__":
    start_sent = "This cafe is a hidden gem! The cozy atmosphere and excellent coffee make it the perfect spot to relax and unwind. The staff are friendly and attentive, always ensuring that your cup is full."
    shaper = SHAPexplainer()
    results = shaper.shap_eval(start_sent, 45)

    for i in range(len(results)):
        plot(results[i])
