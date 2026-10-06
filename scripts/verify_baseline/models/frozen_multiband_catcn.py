"""
FROZEN BASELINE: Multi-Band Cochlear Gammatone + Causal ERP Cross-Attention CA-TCN (v1).
Reference Architecture: arXiv:2603.26394 + Tonotopic Cochlear Cross-Attention.
Status: FROZEN & PRESERVED FOR BENCHMARK REPRODUCIBILITY.

Cohort Benchmark Results across 18 DTU Subjects (Commit 804f419):
  - 5.0s Window 2AFC Accuracy:  65.74% (Matches canonical single-band baseline 65.70%)
  - 10.0s Window 2AFC Accuracy: 71.02%
  - 20.0s Window 2AFC Accuracy: 76.39% (S15=95.8%, S14=91.7%, S13=91.7%, S7=87.5%, S8=87.5%)
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectionalDepthwiseConv1d(nn.Module):
    """
    Depthwise 1D Convolution with strict directional padding.
    - 'causal': Receptive field extends strictly into past (t, t-1, t-2, ...).
    - 'anticausal': Receptive field extends strictly into future (t, t+1, t+2, ...).
    """
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, direction: str = 'causal'):
        super().__init__()
        assert direction in ['causal', 'anticausal'], f"Invalid direction: {direction}"
        self.direction = direction
        self.pad_len = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, 
            dilation=dilation, padding=0, groups=channels, bias=False
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.direction == 'causal':
            x_padded = F.pad(x, (self.pad_len, 0))
        else:
            x_padded = F.pad(x, (0, self.pad_len))
        return self.conv(x_padded)

class DepthwiseSeparableTCNBlock(nn.Module):
    """
    Depthwise-Separable Temporal Convolutional Block with Residual Connection.
    """
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, direction: str = 'causal', dropout: float = 0.2):
        super().__init__()
        self.depthwise = DirectionalDepthwiseConv1d(
            channels, kernel_size=kernel_size, dilation=dilation, direction=direction
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.depthwise(x)
        out = F.elu(self.bn1(out))
        out = self.pointwise(out)
        out = self.bn2(out)
        out = self.dropout(out)
        return F.elu(out + res)

class CATCN_MultiBandAudioEncoder(nn.Module):
    """
    Causal Multi-Band Cochlear Stimulus Encoder.
    Processes N_bands (default 8) Gammatone subbands into hidden representations.
    Strictly CAUSAL with receptive field = 1 + 2 * (1 + 2 + 4 + 8 + 16) = 63 samples (984.4 ms at 64 Hz).
    """
    def __init__(self, in_channels: int = 8, hidden_dim: int = 64, dilations: list[int] = [1, 2, 4, 8, 16], dropout: float = 0.2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        # 1x1 spectral projection from N cochlear subbands to hidden channels
        self.spectral_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn_proj = nn.BatchNorm1d(hidden_dim)
        
        self.blocks = nn.ModuleList([
            DepthwiseSeparableTCNBlock(
                channels=hidden_dim, kernel_size=3, dilation=d, direction='causal', dropout=dropout
            )
            for d in dilations
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_channels, T]
        feat = F.elu(self.bn_proj(self.spectral_proj(x)))
        for block in self.blocks:
            feat = block(feat)
        return feat

class CATCN_EEGEncoder(nn.Module):
    """
    Anticausal EEG Neural Encoder.
    Processes raw multi-channel scalp EEG into hidden representations.
    Strictly ANTICAUSAL with receptive field = 1 + 2 * (1 + 2 + 4) = 15 samples (234.4 ms at 64 Hz).
    """
    def __init__(self, in_channels: int = 64, hidden_dim: int = 64, dilations: list[int] = [1, 2, 4], dropout: float = 0.2):
        super().__init__()
        self.spatial_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn_spatial = nn.BatchNorm1d(hidden_dim)
        
        self.blocks = nn.ModuleList([
            DepthwiseSeparableTCNBlock(
                channels=hidden_dim, kernel_size=3, dilation=d, direction='anticausal', dropout=dropout
            )
            for d in dilations
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_eeg, T]
        feat = F.elu(self.bn_spatial(self.spatial_proj(x)))
        for block in self.blocks:
            feat = block(feat)
        return feat

class CrossCorrelationClassificationHead(nn.Module):
    """
    Normalized Multi-Lag Cross-Correlation Head with Guaranteed Anti-Symmetry.
    Computes normalized cross-correlation across temporal lags tau in [-max_lag_samples, +max_lag_samples],
    then passes the concatenated correlation coefficients to a linear classifier without bias.
    """
    def __init__(self, hidden_dim: int = 64, max_lag_samples: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_lag = max_lag_samples
        self.num_lags = 2 * max_lag_samples + 1
        self.classifier = nn.Linear(hidden_dim * self.num_lags, 1, bias=False)
        
    def compute_cross_correlation(self, z_eeg: torch.Tensor, z_audio: torch.Tensor) -> torch.Tensor:
        B, D, T = z_eeg.shape
        
        # Standardize along temporal dimension (zero mean, unit variance per channel)
        ze_mean = z_eeg.mean(dim=-1, keepdim=True)
        ze_std = z_eeg.std(dim=-1, keepdim=True) + 1e-8
        ze_norm = (z_eeg - ze_mean) / ze_std
        
        za_mean = z_audio.mean(dim=-1, keepdim=True)
        za_std = z_audio.std(dim=-1, keepdim=True) + 1e-8
        za_norm = (z_audio - za_mean) / za_std
        
        corrs = []
        for tau in range(-self.max_lag, self.max_lag + 1):
            if tau > 0:
                ze_slice = ze_norm[:, :, tau:]
                za_slice = za_norm[:, :, :-tau]
            elif tau < 0:
                abs_tau = abs(tau)
                ze_slice = ze_norm[:, :, :-abs_tau]
                za_slice = za_norm[:, :, abs_tau:]
            else:
                ze_slice = ze_norm
                za_slice = za_norm
                
            r_tau = (ze_slice * za_slice).mean(dim=-1) # [B, D]
            corrs.append(r_tau)
            
        r_all = torch.stack(corrs, dim=-1).view(B, -1) # [B, D * num_lags]
        return r_all
        
    def forward(self, z_eeg: torch.Tensor, z_a: torch.Tensor, z_b: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        r_a = self.compute_cross_correlation(z_eeg, z_a)
        r_b = self.compute_cross_correlation(z_eeg, z_b)
        
        logit_a = self.classifier(r_a).squeeze(-1) # [B]
        logit_b = self.classifier(r_b).squeeze(-1) # [B]
        
        delta = logit_a - logit_b
        return delta, (logit_a, logit_b)

class FrozenMultiBandCATCNDecoder(nn.Module):
    """
    Frozen v1 Multi-Band Cochlear Gammatone + CA-TCN Decoder.
    Preserved exactly as evaluated across the 18 DTU subjects with 65.74% 5s cohort accuracy.
    """
    def __init__(
        self,
        eeg_channels: int = 8,
        audio_bands: int = 8,
        hidden_dim: int = 64,
        max_lag_samples: int = 8,
        dropout: float = 0.2
    ):
        super().__init__()
        self.eeg_channels = eeg_channels
        self.audio_bands = audio_bands
        self.hidden_dim = hidden_dim
        
        self.audio_encoder = CATCN_MultiBandAudioEncoder(
            in_channels=audio_bands,
            hidden_dim=hidden_dim,
            dilations=[1, 2, 4, 8, 16],
            dropout=dropout
        )
        self.eeg_encoder = CATCN_EEGEncoder(
            in_channels=eeg_channels,
            hidden_dim=hidden_dim,
            dilations=[1, 2, 4],
            dropout=dropout
        )
        self.classifier_head = CrossCorrelationClassificationHead(
            hidden_dim=hidden_dim,
            max_lag_samples=max_lag_samples
        )
        
    def forward(
        self,
        eeg: torch.Tensor,
        audio_a: torch.Tensor,
        audio_b: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        z_eeg = self.eeg_encoder(eeg)
        z_a = self.audio_encoder(audio_a)
        z_b = self.audio_encoder(audio_b)
        
        delta, (logit_a, logit_b) = self.classifier_head(z_eeg, z_a, z_b)
        return delta, (logit_a, logit_b), (z_eeg, z_a, z_b)
