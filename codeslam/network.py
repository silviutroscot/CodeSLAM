from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .losses import positive_scale
from .proximity import depth_to_proximity, proximity_to_depth
from .types import DepthPrediction


def _group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        linear: bool = False,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=not use_norm),
        ]
        if use_norm:
            layers.append(nn.GroupNorm(_group_count(out_channels), out_channels))
        if not linear:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = ConvBlock(in_channels, out_channels, stride=2, linear=False, use_norm=True)
        self.refine = ConvBlock(out_channels, out_channels, stride=1, linear=False, use_norm=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.down(x))


class ConditioningFusion(nn.Module):
    def __init__(
        self,
        depth_channels: int,
        image_channels: int,
        out_channels: int,
        *,
        linear: bool,
        use_norm: bool,
    ) -> None:
        super().__init__()
        self.image_projection = nn.Conv2d(image_channels, depth_channels, kernel_size=1, bias=True)
        self.mix = ConvBlock(
            depth_channels + image_channels + depth_channels,
            out_channels,
            stride=1,
            linear=linear,
            use_norm=use_norm,
        )

    def forward(self, depth_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        projected = self.image_projection(image_features)
        fused = torch.cat([depth_features, image_features, depth_features * projected], dim=1)
        return self.mix(fused)


class UpsampleStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        linear: bool,
        use_norm: bool,
        use_deconvolution: bool = False,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.use_deconvolution = use_deconvolution
        if use_deconvolution:
            self.deconvolution = nn.ConvTranspose2d(
                in_channels,
                in_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            )
        self.project = ConvBlock(in_channels, out_channels, stride=1, linear=linear, use_norm=use_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_deconvolution:
            x = self.deconvolution(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.project(x)


@dataclass
class DecoderState:
    proximity_pyramid: list[torch.Tensor]
    scale_pyramid: list[torch.Tensor]


class CodeSLAMDepthModel(nn.Module):
    """Conditioned depth auto-encoder described in the CodeSLAM paper."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()

        image_channels = [
            self.config.base_channels,
            self.config.base_channels * 2,
            self.config.base_channels * 4,
            self.config.base_channels * 8,
            self.config.base_channels * 16,
        ]
        self._image_channels = image_channels

        self.image_encoder = nn.ModuleList()
        in_channels = 1
        for out_channels in image_channels:
            self.image_encoder.append(EncoderStage(in_channels, out_channels))
            in_channels = out_channels

        self.depth_encoder = nn.ModuleList()
        self.depth_fusions = nn.ModuleList()
        in_channels = 1
        for out_channels, image_channels_at_level in zip(image_channels, image_channels):
            self.depth_encoder.append(EncoderStage(in_channels, out_channels))
            self.depth_fusions.append(
                ConditioningFusion(out_channels, image_channels_at_level, out_channels, linear=False, use_norm=True)
            )
            in_channels = out_channels

        bottleneck_height = self.config.input_height // 32
        bottleneck_width = self.config.input_width // 32
        self._bottleneck_shape = (image_channels[-1], bottleneck_height, bottleneck_width)
        bottleneck_dim = image_channels[-1] * bottleneck_height * bottleneck_width

        self.posterior_hidden = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bottleneck_dim, self.config.latent_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.posterior_mean = nn.Linear(self.config.latent_hidden_dim, self.config.code_dim)
        self.posterior_logvar = nn.Linear(self.config.latent_hidden_dim, self.config.code_dim)

        self.code_to_bottleneck = nn.Linear(self.config.code_dim, bottleneck_dim)
        self.image_bottleneck_bias = nn.Conv2d(image_channels[-1], image_channels[-1], kernel_size=1, bias=True)

        decoder_channels = [image_channels[-2], image_channels[-3], image_channels[-4], image_channels[-5]]
        skip_channels = [image_channels[-2], image_channels[-3], image_channels[-4], image_channels[-5]]

        self.decoder_up = nn.ModuleList()
        self.decoder_fusions = nn.ModuleList()
        self.proximity_heads = nn.ModuleList()

        decoder_in = image_channels[-1]
        for index, (out_channels, skip_channels_at_level) in enumerate(zip(decoder_channels, skip_channels)):
            self.decoder_up.append(
                UpsampleStage(
                    decoder_in,
                    out_channels,
                    linear=self.config.linear_decoder,
                    use_norm=not self.config.linear_decoder,
                    use_deconvolution=index == len(decoder_channels) - 1,
                )
            )
            self.decoder_fusions.append(
                ConditioningFusion(
                    out_channels,
                    skip_channels_at_level,
                    out_channels,
                    linear=self.config.linear_decoder,
                    use_norm=not self.config.linear_decoder,
                )
            )
            self.proximity_heads.append(nn.Conv2d(out_channels, 1, kernel_size=3, padding=1))
            decoder_in = out_channels

        self.uncertainty_seed = ConvBlock(image_channels[-1], decoder_channels[0], stride=1, linear=False, use_norm=True)
        self.uncertainty_up = nn.ModuleList()
        self.uncertainty_fusions = nn.ModuleList()
        self.uncertainty_heads = nn.ModuleList()

        uncertainty_in = decoder_channels[0]
        for index, (out_channels, skip_channels_at_level) in enumerate(zip(decoder_channels, skip_channels)):
            self.uncertainty_up.append(
                UpsampleStage(
                    uncertainty_in,
                    out_channels,
                    linear=False,
                    use_norm=True,
                    use_deconvolution=index == len(decoder_channels) - 1,
                )
            )
            self.uncertainty_fusions.append(
                ConditioningFusion(out_channels, skip_channels_at_level, out_channels, linear=False, use_norm=True)
            )
            self.uncertainty_heads.append(nn.Conv2d(out_channels, 1, kernel_size=3, padding=1))
            uncertainty_in = out_channels

    def _ensure_grayscale(self, intensity: torch.Tensor) -> torch.Tensor:
        if intensity.ndim == 3:
            intensity = intensity.unsqueeze(0)
        if intensity.shape[1] == 1:
            return intensity
        if intensity.shape[1] == 3:
            r, g, b = intensity[:, 0:1], intensity[:, 1:2], intensity[:, 2:3]
            return 0.2989 * r + 0.5870 * g + 0.1140 * b
        raise ValueError(f"Expected 1 or 3 intensity channels, got {intensity.shape[1]}")

    def encode_image(self, intensity: torch.Tensor) -> list[torch.Tensor]:
        intensity = self._ensure_grayscale(intensity)
        features = []
        x = intensity
        for stage in self.image_encoder:
            x = stage(x)
            features.append(x)
        return features

    def encode_depth(
        self,
        proximity: torch.Tensor,
        image_features: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = proximity
        for stage, fusion, image_feature in zip(self.depth_encoder, self.depth_fusions, image_features):
            x = stage(x)
            x = fusion(x, image_feature)
        posterior_hidden = self.posterior_hidden(x)
        return self.posterior_mean(posterior_hidden), self.posterior_logvar(posterior_hidden)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor, sample_posterior: bool) -> torch.Tensor:
        if not sample_posterior:
            return mean
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode_from_features(self, image_features: list[torch.Tensor], code: torch.Tensor) -> DecoderState:
        batch_size = code.shape[0]
        channels, height, width = self._bottleneck_shape
        x = self.code_to_bottleneck(code).view(batch_size, channels, height, width)
        x = x + self.image_bottleneck_bias(image_features[-1])

        proximity_pyramid: list[torch.Tensor] = []
        skip_features = list(reversed(image_features[:-1]))
        for up, fusion, head, skip in zip(self.decoder_up, self.decoder_fusions, self.proximity_heads, skip_features):
            x = up(x)
            x = fusion(x, skip)
            proximity_pyramid.append(head(x))

        uncertainty = self.uncertainty_seed(image_features[-1])
        scale_pyramid: list[torch.Tensor] = []
        for up, fusion, head, skip in zip(self.uncertainty_up, self.uncertainty_fusions, self.uncertainty_heads, skip_features):
            uncertainty = up(uncertainty)
            uncertainty = fusion(uncertainty, skip)
            scale_pyramid.append(head(uncertainty))

        return DecoderState(proximity_pyramid=proximity_pyramid, scale_pyramid=scale_pyramid)

    def decode(self, intensity: torch.Tensor, code: torch.Tensor) -> DecoderState:
        image_features = self.encode_image(intensity)
        return self.decode_from_features(image_features, code)

    def predict_from_image_features(self, image_features: list[torch.Tensor], code: torch.Tensor) -> DepthPrediction:
        decoded = self.decode_from_features(image_features, code)
        scale_pyramid = [positive_scale(scale, self.config.min_uncertainty) for scale in decoded.scale_pyramid]
        depth_pyramid = [
            proximity_to_depth(proximity, self.config.proximity_transition).clamp(self.config.min_depth, self.config.max_depth)
            for proximity in decoded.proximity_pyramid
        ]
        return DepthPrediction(
            code=code,
            proximity_pyramid=decoded.proximity_pyramid,
            depth_pyramid=depth_pyramid,
            scale_pyramid=scale_pyramid,
            posterior_mean=None,
            posterior_logvar=None,
        )

    def zero_code(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.config.code_dim, device=device, dtype=dtype)

    def forward(
        self,
        intensity: torch.Tensor,
        depth: torch.Tensor | None = None,
        *,
        code: torch.Tensor | None = None,
        sample_posterior: bool = True,
    ) -> DepthPrediction:
        intensity = self._ensure_grayscale(intensity)
        image_features = self.encode_image(intensity)

        posterior_mean = None
        posterior_logvar = None
        if code is None:
            if depth is None:
                code = self.zero_code(intensity.shape[0], intensity.device, intensity.dtype)
            else:
                proximity = depth_to_proximity(depth, self.config.proximity_transition)
                posterior_mean, posterior_logvar = self.encode_depth(proximity, image_features)
                code = self.reparameterize(posterior_mean, posterior_logvar, sample_posterior)

        prediction = self.predict_from_image_features(image_features, code)
        prediction.posterior_mean = posterior_mean
        prediction.posterior_logvar = posterior_logvar
        return prediction

    @torch.no_grad()
    def precompute_linear_jacobian(
        self,
        intensity: torch.Tensor,
        *,
        level: int = -1,
        chunk_size: int = 16,
    ) -> torch.Tensor:
        if not self.config.linear_decoder:
            raise ValueError("Linear Jacobian precomputation is only valid for the linear decoder.")
        intensity = self._ensure_grayscale(intensity)
        if intensity.shape[0] != 1:
            raise ValueError("Jacobian precomputation expects a batch size of 1.")

        image_features = self.encode_image(intensity)
        zero_code = self.zero_code(1, intensity.device, intensity.dtype)
        baseline = self.decode_from_features(image_features, zero_code).proximity_pyramid[level][0].reshape(-1)

        jacobian_columns = []
        identity = torch.eye(self.config.code_dim, device=intensity.device, dtype=intensity.dtype)
        for start in range(0, self.config.code_dim, chunk_size):
            stop = min(start + chunk_size, self.config.code_dim)
            codes = identity[start:stop]
            repeated_features = [feature.repeat(stop - start, 1, 1, 1) for feature in image_features]
            decoded = self.decode_from_features(repeated_features, codes).proximity_pyramid[level]
            chunk = decoded.reshape(stop - start, -1) - baseline.unsqueeze(0)
            jacobian_columns.append(chunk.transpose(0, 1))
        return torch.cat(jacobian_columns, dim=1)
