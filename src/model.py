"""MGCN: a custom multi-scale gated image-captioning model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MultiScaleGatedEncoder(nn.Module):
    """Learns scale weights pi_s and fuses three dilated visual representations."""
    def __init__(self, visual_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(3, 64, stride=2),
            ConvNormAct(64, 128, stride=2),
            ConvNormAct(128, visual_dim, stride=2),
        )
        self.scales = nn.ModuleList([
            nn.Sequential(ConvNormAct(visual_dim, visual_dim, dilation=rate), ConvNormAct(visual_dim, visual_dim, dilation=rate))
            for rate in (1, 2, 4)
        ])
        self.gate = nn.Linear(visual_dim * 3, 3)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.stem(images)
        features = [branch(shared) for branch in self.scales]
        descriptors = torch.cat([feature.mean(dim=(2, 3)) for feature in features], dim=1)
        pi = F.softmax(self.gate(descriptors), dim=1)
        fused = sum(pi[:, index, None, None, None] * feature for index, feature in enumerate(features))
        spatial_features = fused.flatten(2).transpose(1, 2)
        return spatial_features, pi


class VisualEvidenceDecoder(nn.Module):
    """Custom recurrent decoder with an explicit attended-image evidence memory."""
    def __init__(self, vocab_size: int, visual_dim: int = 256, embedding_dim: int = 256, hidden_dim: int = 512, pad_id: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.query = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.feature_projection = nn.Linear(visual_dim, hidden_dim, bias=False)
        self.attention_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attention_score = nn.Linear(hidden_dim, 1, bias=False)
        self.memory_gate = nn.Linear(embedding_dim + visual_dim + hidden_dim, visual_dim)
        self.candidate = nn.Linear(embedding_dim + visual_dim + visual_dim + hidden_dim, hidden_dim)
        self.state_gate = nn.Linear(embedding_dim + visual_dim + visual_dim + hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim + visual_dim + visual_dim, vocab_size)

    def step(self, previous_word: torch.Tensor, features: torch.Tensor, projected_features: torch.Tensor, hidden: torch.Tensor, memory: torch.Tensor):
        embedding = self.embedding(previous_word)
        query = torch.tanh(self.query(torch.cat([embedding, hidden], dim=1)))
        energy = self.attention_score(torch.tanh(projected_features + self.attention_projection(query).unsqueeze(1))).squeeze(-1)
        attention = F.softmax(energy, dim=1)
        context = torch.bmm(attention.unsqueeze(1), features).squeeze(1)

        decoder_input = torch.cat([embedding, context, hidden], dim=1)
        memory_gate = torch.sigmoid(self.memory_gate(decoder_input))
        memory = memory_gate * context + (1.0 - memory_gate) * memory

        state_input = torch.cat([embedding, context, memory, hidden], dim=1)
        candidate = torch.tanh(self.candidate(state_input))
        state_gate = torch.sigmoid(self.state_gate(state_input))
        hidden = state_gate * candidate + (1.0 - state_gate) * hidden
        logits = self.output(torch.cat([hidden, context, memory], dim=1))
        return logits, hidden, memory, attention

    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = features.size(0)
        hidden = features.new_zeros(batch_size, self.hidden_dim)
        memory = features.new_zeros(batch_size, features.size(-1))
        projected_features = self.feature_projection(features)
        logits, attentions = [], []
        for step in range(captions.size(1) - 1):
            output, hidden, memory, attention = self.step(captions[:, step], features, projected_features, hidden, memory)
            logits.append(output)
            attentions.append(attention)
        return torch.stack(logits, dim=1), torch.stack(attentions, dim=1)


class MGCN(nn.Module):
    def __init__(self, vocab_size: int, visual_dim: int = 256, embedding_dim: int = 256, hidden_dim: int = 512, pad_id: int = 0):
        super().__init__()
        self.encoder = MultiScaleGatedEncoder(visual_dim)
        self.decoder = VisualEvidenceDecoder(vocab_size, visual_dim, embedding_dim, hidden_dim, pad_id)

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, scale_weights = self.encoder(images)
        logits, attention = self.decoder(features, captions)
        return logits, attention, scale_weights
