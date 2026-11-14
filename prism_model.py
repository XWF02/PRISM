"""PRISM backbone aligned with Sections 3.2-3.4 of the manuscript."""
from __future__ import annotations

import torch
from torch import nn


class GradientReversal(torch.autograd.Function):
    """Identity in the forward pass; reverse the gradient in the backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float = 1.0):
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.scale * grad_output, None


class SpatialFrequencyAttention(nn.Module):
    """Lightweight channel attention followed by learnable five-band weights.

    Input is flattened MDE with channel-major layout:
    [channel 1: DE(5), dDE(5), channel 2: DE(5), dDE(5), ...].
    The same band weight is applied to the 0th- and 1st-order component.
    """

    def __init__(self, channels: int, bands: int = 5, dropout: float = 0.3):
        super().__init__()
        hidden = max(8, channels // 4)
        self.channel_attention = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )
        self.band_weights = nn.Parameter(torch.ones(bands))
        self.dropout = nn.Dropout(dropout)
        self.channels = channels
        self.bands = bands

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected = self.channels * self.bands * 2
        if x.ndim == 2:
            if x.shape[1] != expected:
                raise ValueError(f"Expected [B,{expected}], received {tuple(x.shape)}")
            x = x.reshape(x.shape[0], self.channels, 2 * self.bands)
        elif x.ndim != 3 or x.shape[1:] != (self.channels, 2 * self.bands):
            raise ValueError(
                f"Expected [B,{expected}] or [B,{self.channels},{2*self.bands}], "
                f"received {tuple(x.shape)}"
            )

        channel_descriptor = x.mean(dim=-1)
        channel_weights = self.channel_attention(channel_descriptor).unsqueeze(-1)
        frequency_weights = self.band_weights.repeat(2).view(1, 1, 2 * self.bands)
        return self.dropout(x * channel_weights * frequency_weights).flatten(1)


class ResidualMLPBlock(nn.Module):
    """Residual projection block; the manuscript specifies residual blocks."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.shortcut = (
            nn.Identity() if in_features == out_features
            else nn.Linear(in_features, out_features)
        )
        self.output_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_activation(self.main(x) + self.shortcut(x))


class PRISM(nn.Module):
    """PRISM for a fixed candidate K.

    The manuscript's bottleneck dimensions are 620->512->256->256 for
    SEED/SEED-IV and 320->512->256->256 for FACED. The final bottleneck
    representation is therefore 256-D; no additional 256->256 projection
    is inserted after the three stated residual blocks.
    """

    def __init__(
        self,
        channels: int,
        num_classes: int,
        num_domains: int,
        bands: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        if num_domains < num_classes:
            raise ValueError("K must be >= the number of emotion classes")
        self.channels = channels
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.input_dim = channels * bands * 2
        self.bands = bands

        self.attention = SpatialFrequencyAttention(channels, bands, dropout)
        self.bottleneck = nn.Sequential(
            ResidualMLPBlock(self.input_dim, 512, dropout),
            ResidualMLPBlock(512, 256, dropout),
            ResidualMLPBlock(256, 256, dropout),
        )
        self.emotion_classifier = nn.Linear(256, num_classes)
        self.domain_class_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes * num_domains),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_domains),
        )

    def forward(self, x: torch.Tensor, grl_scale: float = 1.0):
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected MDE tensor [B,{self.input_dim}], got {tuple(x.shape)}")
        attended = self.attention(x)
        z = self.bottleneck(attended)
        emotion_logits = self.emotion_classifier(z)
        joint_domain_class_logits = self.domain_class_classifier(z)
        # GRL is applied only to the latent-domain adversarial branch.
        domain_logits = self.domain_classifier(GradientReversal.apply(z, grl_scale))
        return emotion_logits, joint_domain_class_logits, domain_logits, z

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[0]
