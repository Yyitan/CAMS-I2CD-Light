import torch
from torch import nn
import torch.nn.functional as F


class ResidualProjector(nn.Module):
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


class I2CDLight(nn.Module):
    """A minimal disentangle-compose-redisentangle auxiliary head."""

    def __init__(self, dim, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.i2cd_attr_disentangler = ResidualProjector(dim, hidden_dim, dropout)
        self.i2cd_obj_disentangler = ResidualProjector(dim, hidden_dim, dropout)
        self.i2cd_composer = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def compose(self, attr_feature, obj_feature):
        composed = self.i2cd_composer(torch.cat([attr_feature, obj_feature], dim=-1))
        return F.normalize(composed, p=2, dim=-1)

    def forward(self, com_feature):
        attr_feature = self.i2cd_attr_disentangler(com_feature)
        obj_feature = self.i2cd_obj_disentangler(com_feature)
        recomposed_feature = self.compose(attr_feature, obj_feature)
        attr_cycle = self.i2cd_attr_disentangler(recomposed_feature)
        obj_cycle = self.i2cd_obj_disentangler(recomposed_feature)
        return attr_feature, obj_feature, recomposed_feature, attr_cycle, obj_cycle
