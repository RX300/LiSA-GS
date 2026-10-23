"""Shared point-light shading and the transport method interface."""

from abc import ABC, abstractmethod

import torch
from torch import nn


def direction_encoding(direction, bands=4):
    frequency = 2 ** torch.arange(bands, device=direction.device, dtype=direction.dtype)
    phase = direction[..., None] * frequency * torch.pi
    return torch.cat((direction, phase.sin().flatten(-2), phase.cos().flatten(-2)), dim=-1)


class TransportBase(nn.Module, ABC):
    """Linear-radiance interface used by LiSA.

    Subclasses declare constructor defaults and implement linear pixel shading.
    Shared scene geometry, observations and optimizers remain outside this class.
    """
    defaults = dict(feature_dim=32)
    geometry = "3dgs"
    cli_fields = ("feature_dim",)

    def __init__(self, light_scale=1.0):
        super().__init__()
        self.register_buffer("light_scale", torch.tensor(float(light_scale)))

    @abstractmethod
    def forward(self, gaussians, receivers, eye, light_pos, light_intensity,
                source_visibility, port_active=True):
        """Return [M, 3] linear foreground radiance for the pixel receivers."""


