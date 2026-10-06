"""
Multi-Band Cochlear Gammatone + Causal ERP Cross-Attention CA-TCN (Sinc-CATCN v2).
Reference Architecture: arXiv:2603.26394 + Biological SincNet Front-End + Bilinear Cross-Attention.

Key Innovations:
1. Biological SincNet Filterbank (SincConvEEG):
   Parameterized 1D sinc temporal bandpass filters initialized to 8 canonical EEG neural
   rhythms (Delta 0.5-4 Hz, Theta 4-8 Hz, Alpha 8-13 Hz, Beta 13-30 Hz) with Hamming windowing.
   Decomposes scalp EEG into true neural oscillation bands matching cochlear tonotopic subbands.
2. Spatial-Spectral Beamformer:
   Depthwise spatial combination mixing multi-channel electrodes across biological rhythms
   to extract auditory cortical dipole sources (Heschl's gyrus / Superior Temporal Gyrus).
3. Bilinear Multi-Band Cross-Attention Head:
   Cross-spectral latent projection matching acoustic subbands with cortical neural rhythms.
4. Physiological Causal ERP Latency Window:
   Focused latency window tau in [-2, +18] samples (-31 ms to +281 ms), directly targeting
   the cortical N100 and P200 auditory evoked potentials while suppressing non-physiological noise.
5. Strict Anti-Symmetry & Zero Shortcut Leakage:
   Delta(A, B) = -Delta(B, A) mathematically exact. No independent audio feature shortcuts.
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

class SincConvEEG(nn.Module):
    """
    Parameterized Biological SincNet Filterbank for Scalp EEG.
    Extracts 8 physiological neural oscillation bands (Delta, Theta, Alpha, Beta)
    directly parameterized by learnable lower cutoff frequencies and bandwidths.
    """
    def __init__(self, out_bands: int = 8, kernel_size: int = 65, sample_rate: float = 64.0):
        super().__init__()
        self.out_bands = out_bands
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2.0
        
        # 8 Canonical Biological EEG Bands:
        # 1. Low Delta (0.5 - 2.0 Hz) - Phrase/prosodic tracking
        # 2. High Delta (2.0 - 4.0 Hz) - Syllable grouping
        # 3. Low Theta (4.0 - 6.0 Hz) - Acoustic syllable boundaries
        # 4. High Theta (6.0 - 8.0 Hz) - Phonemic envelope rate
        # 5. Low Alpha (8.0 - 10.5 Hz) - Auditory attentional gating
        # 6. High Alpha (10.5 - 13.0 Hz) - Parietal alpha suppression
        # 7. Low Beta (13.0 - 20.0 Hz) - Temporal prediction
        # 8. Mid Beta (20.0 - 30.0 Hz) - Motor/auditory integration
        f1_init_hz = torch.tensor([0.5, 2.0, 4.0, 6.0, 8.0, 10.5, 13.0, 20.0])
        f2_init_hz = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.5, 13.0, 20.0, 30.0])
        band_init_hz = f2_init_hz - f1_init_hz
        
        self.min_low_hz = 0.2
        self.min_band_hz = 1.0
        
        f1_norm = (f1_init_hz - self.min_low_hz) / (self.nyquist - self.min_low_hz - self.min_band_hz + 1e-6)
        f1_norm = torch.clamp(f1_norm, 1e-4, 1.0 - 1e-4)
        self.f1_raw = nn.Parameter(torch.log(f1_norm / (1.0 - f1_norm)))
        
        band_norm = (band_init_hz - self.min_band_hz) / (self.nyquist - f1_init_hz - self.min_band_hz + 1e-6)
        band_norm = torch.clamp(band_norm, 1e-4, 1.0 - 1e-4)
        self.band_raw = nn.Parameter(torch.log(band_norm / (1.0 - band_norm)))
        
        # Symmetric Hamming window
        n = torch.linspace(0, self.kernel_size - 1, self.kernel_size)
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * n / (self.kernel_size - 1))
        self.register_buffer('window', window)
        
        t_right = torch.linspace(1, (self.kernel_size - 1) / 2, steps=int((self.kernel_size - 1) / 2))
        self.register_buffer('t_right', t_right)
        
    def get_bands(self):
        range_f1 = self.nyquist - self.min_low_hz - self.min_band_hz
        f1 = self.min_low_hz + range_f1 * torch.sigmoid(self.f1_raw)
        range_band = self.nyquist - f1 - self.min_band_hz
        band = self.min_band_hz + range_band * torch.sigmoid(self.band_raw)
        return f1 / self.sample_rate, (f1 + band) / self.sample_rate, band / self.sample_rate
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        B, C, T = x.shape
        f1, f2, band = self.get_bands()
        t = self.t_right
        
        f1_2pi_t = 2 * math.pi * f1.view(-1, 1) * t.view(1, -1)
        f2_2pi_t = 2 * math.pi * f2.view(-1, 1) * t.view(1, -1)
        
        low_pass1 = torch.sin(f1_2pi_t) / (math.pi * t.view(1, -1))
        low_pass2 = torch.sin(f2_2pi_t) / (math.pi * t.view(1, -1))
        
        band_pass_right = low_pass2 - low_pass1
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_left = torch.flip(band_pass_right, dims=[1])
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass * self.window.view(1, -1)
        
        filters = band_pass.view(self.out_bands, 1, 1, self.kernel_size)
        pad = (self.kernel_size - 1) // 2
        # Apply filterbank across all electrodes simultaneously: [B, 1, C, T] -> [B, out_bands, C, T]
        out = F.conv2d(x.unsqueeze(1), filters, padding=(0, pad))
        return out.view(B, self.out_bands * C, T)

class SincCATCN_EEGEncoder(nn.Module):
    """
    Biological Sinc-Anticausal EEG Neural Encoder.
    Applies Sinc filterbank to decompose each electrode into 8 biological rhythms,
    followed by spatial dipole beamforming and anticausal TCN blocks.
    Receptive Field: ~234 ms future anticausal + 1015 ms biological sinc context.
    """
    def __init__(
        self,
        in_channels: int = 8,
        hidden_dim: int = 64,
        sinc_bands: int = 8,
        dilations: list[int] = [1, 2, 4],
        dropout: float = 0.2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.sinc_bands = sinc_bands
        self.sinc_filter = SincConvEEG(out_bands=sinc_bands, kernel_size=65, sample_rate=64.0)
        self.spatial_proj = nn.Conv1d(in_channels * sinc_bands, hidden_dim, kernel_size=1, bias=False)
        self.bn_spatial = nn.BatchNorm1d(hidden_dim)
        
        self.blocks = nn.ModuleList([
            DepthwiseSeparableTCNBlock(
                channels=hidden_dim, kernel_size=3, dilation=d, direction='anticausal', dropout=dropout
            )
            for d in dilations
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_channels, T]
        sinc_feats = self.sinc_filter(x) # [B, in_channels * sinc_bands, T]
        feat = F.elu(self.bn_spatial(self.spatial_proj(sinc_feats)))
        for block in self.blocks:
            feat = block(feat)
        return feat

class SincCATCN_AudioEncoder(nn.Module):
    """
    Causal Multi-Band Cochlear Stimulus Encoder with Bilinear Latent Projection.
    Processes N_bands (Broadband + 8 Gammatone subbands) into hidden cortical representations.
    Strictly CAUSAL with receptive field = 63 samples (984.4 ms at 64 Hz).
    """
    def __init__(
        self,
        in_channels: int = 9,
        hidden_dim: int = 64,
        dilations: list[int] = [1, 2, 4, 8, 16],
        dropout: float = 0.2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        self.spectral_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn_proj = nn.BatchNorm1d(hidden_dim)
        
        self.blocks = nn.ModuleList([
            DepthwiseSeparableTCNBlock(
                channels=hidden_dim, kernel_size=3, dilation=d, direction='causal', dropout=dropout
            )
            for d in dilations
        ])
        
        # Bilinear cross-channel latent projection from Audio into Cortical EEG space
        self.latent_proj = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False)
        self.bn_latent = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = F.elu(self.bn_proj(self.spectral_proj(x)))
        for block in self.blocks:
            feat = block(feat)
        out = F.elu(self.bn_latent(self.latent_proj(feat)))
        return out

class BilinearCrossCorrelationHead(nn.Module):
    """
    Normalized Multi-Lag Cross-Correlation Head with Guaranteed Anti-Symmetry.
    Supports:
    1. 'factored' (Default): Low-rank channel (D=64) + physiological ERP latency kernel (K=21) + learnable scale.
       Total: 86 parameters. Prevents 5.0s window memorization and generalization gaps.
    2. 'linear': Fully unconstrained linear mapping across D * K lags (1,344 parameters).
    
    Guarantees:
    1. Zero Data Leakage: Purely bilinear dot-product between temporally standardized EEG and Audio.
    2. Strict Anti-Symmetry: Delta(A, B) = -Delta(B, A).
    3. Physiological ERP Focus: Focuses on physiological causal delays (N100, P200).
    """
    def __init__(self, hidden_dim: int = 64, min_lag: int = -2, max_lag: int = 18, head_type: str = "factored"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.num_lags = max_lag - min_lag + 1
        self.head_type = head_type
        
        if head_type == "factored":
            self.channel_weight = nn.Parameter(torch.ones(hidden_dim) / math.sqrt(hidden_dim))
            # Gaussian ERP latency prior centered at lag 8 (~125 ms / N100-P200 peak)
            lags = torch.arange(min_lag, max_lag + 1).float()
            prior = torch.exp(-0.5 * ((lags - 8.0) / 4.0) ** 2)
            self.lag_weight = nn.Parameter(prior / prior.sum())
            self.scale = nn.Parameter(torch.tensor(5.0))
        else:
            self.classifier = nn.Linear(hidden_dim * self.num_lags, 1, bias=False)
        
    def compute_cross_correlation(self, z_eeg: torch.Tensor, z_audio: torch.Tensor) -> torch.Tensor:
        B, D, T = z_eeg.shape
        
        ze_mean = z_eeg.mean(dim=-1, keepdim=True)
        ze_std = z_eeg.std(dim=-1, keepdim=True) + 1e-8
        ze_norm = (z_eeg - ze_mean) / ze_std
        
        za_mean = z_audio.mean(dim=-1, keepdim=True)
        za_std = z_audio.std(dim=-1, keepdim=True) + 1e-8
        za_norm = (z_audio - za_mean) / za_std
        
        corrs = []
        for tau in range(self.min_lag, self.max_lag + 1):
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
            
        if self.head_type == "factored":
            R = torch.stack(corrs, dim=1) # [B, num_lags, D]
            weighted_ch = (R * self.channel_weight.view(1, 1, D)).sum(dim=-1) # [B, num_lags]
            score = (weighted_ch * self.lag_weight.view(1, -1)).sum(dim=-1) # [B]
            return self.scale * score
        else:
            r_all = torch.stack(corrs, dim=-1).view(B, -1) # [B, D * num_lags]
            return self.classifier(r_all).squeeze(-1)
        
    def forward(self, z_eeg: torch.Tensor, z_a: torch.Tensor, z_b: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        logit_a = self.compute_cross_correlation(z_eeg, z_a)
        logit_b = self.compute_cross_correlation(z_eeg, z_b)
        
        delta = logit_a - logit_b
        return delta, (logit_a, logit_b)

class SincMultiBandCATCNDecoder(nn.Module):
    """
    Sinc-Biologic Multi-Band Cochlear Gammatone + CA-TCN Decoder (v2).
    Integrates 8-band Biological SincNet EEG filtering, Causal multi-band cochlear TCN,
    and Bilinear multi-lag ERP cross-correlation classification.
    """
    def __init__(
        self,
        eeg_channels: int = 8,
        audio_bands: int = 9,
        hidden_dim: int = 64,
        sinc_bands: int = 8,
        min_lag: int = -2,
        max_lag: int = 18,
        head_type: str = "factored",
        dropout: float = 0.2
    ):
        super().__init__()
        self.eeg_channels = eeg_channels
        self.audio_bands = audio_bands
        self.hidden_dim = hidden_dim
        
        self.eeg_encoder = SincCATCN_EEGEncoder(
            in_channels=eeg_channels,
            hidden_dim=hidden_dim,
            sinc_bands=sinc_bands,
            dilations=[1, 2, 4],
            dropout=dropout
        )
        self.audio_encoder = SincCATCN_AudioEncoder(
            in_channels=audio_bands,
            hidden_dim=hidden_dim,
            dilations=[1, 2, 4, 8, 16],
            dropout=dropout
        )
        self.classifier_head = BilinearCrossCorrelationHead(
            hidden_dim=hidden_dim,
            min_lag=min_lag,
            max_lag=max_lag,
            head_type=head_type
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

# Legacy v1 baseline classes preserved for full backward compatibility
try:
    from models.frozen_multiband_catcn import (
        CATCN_EEGEncoder as CATCN_EEGEncoder_Baseline,
        CATCN_MultiBandAudioEncoder as CATCN_MultiBandAudioEncoder_Baseline,
        CrossCorrelationClassificationHead as CrossCorrelationClassificationHead_Baseline,
        FrozenMultiBandCATCNDecoder as MultiBandCATCNDecoder_Baseline
    )
except ImportError:
    try:
        from .frozen_multiband_catcn import (
            CATCN_EEGEncoder as CATCN_EEGEncoder_Baseline,
            CATCN_MultiBandAudioEncoder as CATCN_MultiBandAudioEncoder_Baseline,
            CrossCorrelationClassificationHead as CrossCorrelationClassificationHead_Baseline,
            FrozenMultiBandCATCNDecoder as MultiBandCATCNDecoder_Baseline
        )
    except ImportError:
        from frozen_multiband_catcn import (
            CATCN_EEGEncoder as CATCN_EEGEncoder_Baseline,
            CATCN_MultiBandAudioEncoder as CATCN_MultiBandAudioEncoder_Baseline,
            CrossCorrelationClassificationHead as CrossCorrelationClassificationHead_Baseline,
            FrozenMultiBandCATCNDecoder as MultiBandCATCNDecoder_Baseline
        )

class MultiBandCATCNDecoder(nn.Module):
    """
    Unified MultiBand CA-TCN Decoder interface.
    Instantiates SincMultiBandCATCNDecoder by default (use_sinc=True)
    or FrozenMultiBandCATCNDecoder (use_sinc=False) for exact legacy reproducibility.
    """
    def __init__(
        self,
        eeg_channels: int = 8,
        audio_bands: int = 8,
        hidden_dim: int = 64,
        max_lag_samples: int = 8,
        min_lag_samples: int = -2,
        head_type: str = "factored",
        dropout: float = 0.2,
        use_sinc: bool = True
    ):
        super().__init__()
        self.use_sinc = use_sinc
        if use_sinc:
            self.model = SincMultiBandCATCNDecoder(
                eeg_channels=eeg_channels,
                audio_bands=audio_bands,
                hidden_dim=hidden_dim,
                min_lag=min_lag_samples,
                max_lag=max_lag_samples,
                head_type=head_type,
                dropout=dropout
            )
        else:
            self.model = MultiBandCATCNDecoder_Baseline(
                eeg_channels=eeg_channels,
                audio_bands=audio_bands,
                hidden_dim=hidden_dim,
                max_lag_samples=max_lag_samples,
                dropout=dropout
            )
            
    # Forward properties to internal model for adaptation access
    @property
    def eeg_encoder(self):
        return self.model.eeg_encoder
        
    @property
    def audio_encoder(self):
        return self.model.audio_encoder
        
    @property
    def classifier_head(self):
        return self.model.classifier_head
        
    def forward(self, eeg, audio_a, audio_b):
        return self.model(eeg, audio_a, audio_b)

def print_summary():
    model = MultiBandCATCNDecoder(eeg_channels=8, audio_bands=9, hidden_dim=64, use_sinc=True)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SincMultiBand-CATCN (8ch EEG, 9-band Audio) Parameter Count: {params:,}")
    model.eval()
    dummy_eeg = torch.randn(2, 8, 320)
    dummy_a = torch.randn(2, 9, 320)
    dummy_b = torch.randn(2, 9, 320)
    delta, (la, lb), (ze, za, zb) = model(dummy_eeg, dummy_a, dummy_b)
    print(f"Output shapes: delta={delta.shape}, la={la.shape}, ze={ze.shape}, za={za.shape}")
    delta_rev, _, _ = model(dummy_eeg, dummy_b, dummy_a)
    diff = torch.max(torch.abs(delta + delta_rev)).item()
    print(f"Anti-symmetry check in eval mode (max |delta(A,B) + delta(B,A)|): {diff:.2e}")

if __name__ == "__main__":
    print_summary()
