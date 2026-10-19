import math
import torch
from torch import nn
import torch.nn.functional as F


class ResidualMLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return F.normalize(x + self.net(self.norm(x)), p=2, dim=-1)


class PromptI2CD(nn.Module):
    """Prompt-space disentangle-compose-redisentangle module."""

    def __init__(self, dim, hidden_dim=512, dropout=0.1, mix_init=0.2):
        super().__init__()
        mix_init = min(max(mix_init, 1e-4), 1 - 1e-4)
        self.prompt_i2cd_mix_logit = nn.Parameter(
            torch.tensor(math.log(mix_init / (1 - mix_init)), dtype=torch.float32)
        )
        self.prompt_i2cd_attr_disentangler = ResidualMLP(dim, hidden_dim, dropout)
        self.prompt_i2cd_obj_disentangler = ResidualMLP(dim, hidden_dim, dropout)
        self.prompt_i2cd_composer = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def compose(self, attr_prompt, obj_prompt):
        composed = self.prompt_i2cd_composer(torch.cat([attr_prompt, obj_prompt], dim=-1))
        return F.normalize(composed, p=2, dim=-1)

    def forward(self, com_prompt, attr_prompt, obj_prompt, pair_idx):
        pair_idx = pair_idx.to(com_prompt.device)
        pair_attr_prompt = attr_prompt[pair_idx[:, 0]]
        pair_obj_prompt = obj_prompt[pair_idx[:, 1]]

        disentangled_attr = self.prompt_i2cd_attr_disentangler(com_prompt)
        disentangled_obj = self.prompt_i2cd_obj_disentangler(com_prompt)
        composed_prompt = self.compose(pair_attr_prompt, pair_obj_prompt)
        redist_attr = self.prompt_i2cd_attr_disentangler(composed_prompt)
        redist_obj = self.prompt_i2cd_obj_disentangler(composed_prompt)

        mix = torch.sigmoid(self.prompt_i2cd_mix_logit).to(com_prompt.dtype)
        refined_prompt = F.normalize((1 - mix) * com_prompt + mix * composed_prompt, p=2, dim=-1)

        return {
            "refined_prompt": refined_prompt,
            "pair_attr_prompt": pair_attr_prompt,
            "pair_obj_prompt": pair_obj_prompt,
            "disentangled_attr": disentangled_attr,
            "disentangled_obj": disentangled_obj,
            "composed_prompt": composed_prompt,
            "redist_attr": redist_attr,
            "redist_obj": redist_obj,
            "mix": mix,
        }
