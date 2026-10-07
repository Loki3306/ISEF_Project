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
        
        # 8 High-Resolution Biological Auditory Tracking Bands in 1.0 - 6.5 Hz:
        # Resolves fine-grained cortical envelope tracking rhythms across the active speech passband:
        # Band 1: 1.00 - 1.60 Hz (Phrase / prosody tracking)
        # Band 2: 1.60 - 2.30 Hz (Slow syllabic grouping)
        # Band 3: 2.30 - 3.00 Hz (Word stress envelope)
        # Band 4: 3.00 - 3.70 Hz (Mean syllable rate)
        # Band 5: 3.70 - 4.40 Hz (Conversational syllabic peak)
        # Band 6: 4.40 - 5.10 Hz (Phonemic boundary transitions)
        # Band 7: 5.10 - 5.80 Hz (Fast syllable rate)
        # Band 8: 5.80 - 6.50 Hz (Upper syllabic modulation)
        f1_init_hz = torch.tensor([1.00, 1.60, 2.30, 3.00, 3.70, 4.40, 5.10, 5.80])
        f2_init_hz = torch.tensor([1.60, 2.30, 3.00, 3.70, 4.40, 5.10, 5.80, 6.50])
        band_init_hz = f2_init_hz - f1_init_hz
        
        self.min_low_hz = 0.5
        self.min_band_hz = 0.3
        
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
    def __init__(
        self,
        hidden_dim: int = 64,
        min_lag: int = -2,
        max_lag: int = 18,
        head_type: str = "linear",
        head_dropout: float = 0.25
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.num_lags = max_lag - min_lag + 1
        self.head_type = head_type
        self.dropout = nn.Dropout(head_dropout)
        
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
            r_all = self.dropout(r_all)
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
        head_type: str = "linear",
        dropout: float = 0.2,
        head_dropout: float = 0.25
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
            head_type=head_type,
            head_dropout=head_dropout
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

# =========================================================================
# V3 ARCHITECTURE: MSCA-CATCN (Multi-Scale Cross-Attention Conformer)
# =========================================================================

class BilateralSpatialBeamformer(nn.Module):
    """
    Near-Ear Bilateral Dipole Spatial Beamformer.
    Extracts 4 differential dipole signals across symmetric left-right electrode pairs:
    (T7, T8), (TP7, TP8), (CP5, CP6), (FC5, FC6).
    Appends the 4 dipoles to the 8 raw monopolar channels (total 12 channels)
    and decomposes all 12 channels through Biological SincNet filterbanks.
    """
    def __init__(self, in_channels: int = 8, sinc_bands: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
        self.sinc_filter = SincConvEEG(out_bands=sinc_bands, kernel_size=65, sample_rate=64.0)
        total_in = sinc_bands * (in_channels + len(self.pairs))
        self.spatial_proj = nn.Conv1d(total_in, hidden_dim, kernel_size=1, bias=False)
        self.bn_spatial = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dipoles = [x[:, i:i+1, :] - x[:, j:j+1, :] for (i, j) in self.pairs]
        x_aug = torch.cat([x] + dipoles, dim=1) # [B, 12, T]
        sinc_feats = self.sinc_filter(x_aug) # [B, sinc_bands * 12, T]
        return F.elu(self.bn_spatial(self.spatial_proj(sinc_feats)))

class MultiScaleDirectionalDepthwiseConv1d(nn.Module):
    """
    Tri-scale directional depthwise convolution (kernels 3, 7, 11).
    Captures phonemic (47 ms), syllabic (109 ms), and word/prosodic (172 ms)
    temporal dynamics in parallel without extra parameter bloat.
    """
    def __init__(self, channels: int, kernels=[3, 7, 11], dilation: int = 1, direction: str = 'causal'):
        super().__init__()
        self.direction = direction
        self.branches = nn.ModuleList()
        c_per = channels // len(kernels)
        self.c_per = c_per
        self.rem = channels - c_per * len(kernels)
        for i, k in enumerate(kernels):
            c_branch = c_per + (self.rem if i == len(kernels)-1 else 0)
            pad = (k - 1) * dilation
            self.branches.append(nn.Sequential(
                nn.ConstantPad1d((pad, 0) if direction == 'causal' else (0, pad), 0.0),
                nn.Conv1d(c_branch, c_branch, kernel_size=k, dilation=dilation, groups=c_branch, bias=False)
            ))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        splits = [self.c_per] * (len(self.branches) - 1) + [self.c_per + self.rem]
        xs = torch.split(x, splits, dim=1)
        outs = [b(chunk) for b, chunk in zip(self.branches, xs)]
        return torch.cat(outs, dim=1)

class MultiScaleDepthwiseSeparableTCNBlock(nn.Module):
    def __init__(self, channels: int, kernels=[3, 7, 11], dilation: int = 1, direction: str = 'causal', dropout: float = 0.2):
        super().__init__()
        self.depthwise = MultiScaleDirectionalDepthwiseConv1d(channels, kernels=kernels, dilation=dilation, direction=direction)
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

class MSCA_EEGEncoder(nn.Module):
    def __init__(self, in_channels=8, hidden_dim=64, sinc_bands=8, dilations=[1, 2, 4], dropout=0.2):
        super().__init__()
        self.beamformer = BilateralSpatialBeamformer(in_channels, sinc_bands, hidden_dim)
        self.blocks = nn.ModuleList([
            MultiScaleDepthwiseSeparableTCNBlock(hidden_dim, kernels=[3, 7, 11], dilation=d, direction='anticausal', dropout=dropout)
            for d in dilations
        ])
    def forward(self, x):
        feat = self.beamformer(x)
        for block in self.blocks:
            feat = block(feat)
        return feat

class MSCA_AudioEncoder(nn.Module):
    def __init__(self, in_channels=9, hidden_dim=64, dilations=[1, 2, 4, 8, 16], dropout=0.2):
        super().__init__()
        # Tonotopic Squeeze-and-Excitation Subband Attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 4),
            nn.ELU(),
            nn.Linear(4, in_channels),
            nn.Sigmoid()
        )
        self.spectral_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn_proj = nn.BatchNorm1d(hidden_dim)
        self.blocks = nn.ModuleList([
            MultiScaleDepthwiseSeparableTCNBlock(hidden_dim, kernels=[3, 7, 11], dilation=d, direction='causal', dropout=dropout)
            for d in dilations
        ])
        self.latent_proj = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False)
        self.bn_latent = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x):
        w = self.se(x).unsqueeze(-1)
        feat = F.elu(self.bn_proj(self.spectral_proj(x * w)))
        for block in self.blocks:
            feat = block(feat)
        return F.elu(self.bn_latent(self.latent_proj(feat)))

class AttentionCrossCorrelationHead(nn.Module):
    """
    Energy-Aware Attention Cross-Correlation Head with Guaranteed Anti-Symmetry.
    Dynamically weights salient acoustic peaks and downweights pauses without stream bias.
    """
    def __init__(self, hidden_dim=64, min_lag=-2, max_lag=18, head_dropout=0.25):
        super().__init__()
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.num_lags = max_lag - min_lag + 1
        self.attn_net = nn.Sequential(
            nn.Conv1d(hidden_dim, 16, kernel_size=1),
            nn.ELU(),
            nn.Conv1d(16, 1, kernel_size=1)
        )
        self.classifier = nn.Linear(hidden_dim * self.num_lags, 1, bias=False)
        self.dropout = nn.Dropout(head_dropout)
        
    def compute_stream_score(self, z_eeg: torch.Tensor, z_audio: torch.Tensor) -> torch.Tensor:
        B, D, T = z_eeg.shape
        ze_norm = (z_eeg - z_eeg.mean(dim=-1, keepdim=True)) / (z_eeg.std(dim=-1, keepdim=True) + 1e-8)
        za_norm = (z_audio - z_audio.mean(dim=-1, keepdim=True)) / (z_audio.std(dim=-1, keepdim=True) + 1e-8)
        
        attn_logits = self.attn_net(z_audio)
        attn_weights = F.softmax(attn_logits, dim=-1)
        
        corrs = []
        for tau in range(self.min_lag, self.max_lag + 1):
            if tau > 0:
                ze_s = ze_norm[:, :, tau:]
                za_s = za_norm[:, :, :-tau]
                w_s = attn_weights[:, :, :-tau]
            elif tau < 0:
                ze_s = ze_norm[:, :, :tau]
                za_s = za_norm[:, :, -tau:]
                w_s = attn_weights[:, :, -tau:]
            else:
                ze_s = ze_norm
                za_s = za_norm
                w_s = attn_weights
                
            w_norm = w_s / (w_s.sum(dim=-1, keepdim=True) + 1e-8)
            r_tau = (ze_s * za_s * w_norm).sum(dim=-1)
            corrs.append(r_tau)
            
        r_all = self.dropout(torch.stack(corrs, dim=-1).view(B, -1))
        return self.classifier(r_all).squeeze(-1)
        
    def forward(self, z_eeg: torch.Tensor, z_a: torch.Tensor, z_b: torch.Tensor):
        sa = self.compute_stream_score(z_eeg, z_a)
        sb = self.compute_stream_score(z_eeg, z_b)
        return sa - sb, (sa, sb)

class MSCAMultiBandCATCNDecoder(nn.Module):
    """
    Multi-Scale Bilateral Sinc-Conformer CA-TCN Decoder (v3).
    Integrates bilateral dipole beamforming, multi-scale temporal convolutions,
    tonotopic subband SE attention, and energy-aware cross-correlation.
    """
    def __init__(
        self,
        eeg_channels: int = 8,
        audio_bands: int = 9,
        hidden_dim: int = 64,
        sinc_bands: int = 8,
        min_lag: int = -2,
        max_lag: int = 18,
        dropout: float = 0.2,
        head_dropout: float = 0.25
    ):
        super().__init__()
        self.eeg_encoder = MSCA_EEGEncoder(eeg_channels, hidden_dim, sinc_bands, [1, 2, 4], dropout)
        self.audio_encoder = MSCA_AudioEncoder(audio_bands, hidden_dim, [1, 2, 4, 8, 16], dropout)
        self.classifier_head = AttentionCrossCorrelationHead(hidden_dim, min_lag, max_lag, head_dropout)
        
    def forward(self, eeg, audio_a, audio_b):
        ze = self.eeg_encoder(eeg)
        za = self.audio_encoder(audio_a)
        zb = self.audio_encoder(audio_b)
        delta, (logit_a, logit_b) = self.classifier_head(ze, za, zb)
        return delta, (logit_a, logit_b), (ze, za, zb)

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

try:
    from models.neuro_conformer import NeuroConformerDecoder
except ImportError:
    try:
        from .neuro_conformer import NeuroConformerDecoder
    except ImportError:
        from neuro_conformer import NeuroConformerDecoder

class MultiBandCATCNDecoder(nn.Module):
    """
    Unified MultiBand CA-TCN Decoder interface.
    Supports:
    - arch='conformer' / 'neuroconformer': Dual-Stream Cross-Modal Neuro-Conformer (v4)
    - arch='msca': Multi-Scale Bilateral Sinc-Conformer (v3)
    - arch='sinc' / use_sinc=True: Sinc-CATCN (v2)
    - arch='baseline' / use_sinc=False: Legacy Baseline (v1)
    """
    def __init__(
        self,
        eeg_channels: int = 8,
        audio_bands: int = 8,
        hidden_dim: int = 64,
        max_lag_samples: int = 8,
        min_lag_samples: int = -2,
        head_type: str = "linear",
        dropout: float = 0.2,
        head_dropout: float = 0.25,
        use_sinc: bool = True,
        arch: str = "sinc",
        subsample_stride: int = 2
    ):
        super().__init__()
        self.use_sinc = use_sinc
        self.arch = arch
        self.subsample_stride = subsample_stride
        if arch in ["conformer", "neuroconformer"]:
            d_m = hidden_dim if hidden_dim >= 64 else 80
            self.model = NeuroConformerDecoder(
                eeg_channels=eeg_channels,
                audio_bands=audio_bands,
                d_model=d_m,
                conformer_blocks=2,
                num_heads=4,
                ffn_dim=d_m * 2,
                min_lag=min_lag_samples,
                max_lag=max_lag_samples,
                dropout=dropout,
                head_dropout=head_dropout,
                subsample_stride=subsample_stride
            )
        elif arch == "msca":
            self.model = MSCAMultiBandCATCNDecoder(
                eeg_channels=eeg_channels,
                audio_bands=audio_bands,
                hidden_dim=hidden_dim,
                sinc_bands=8,
                min_lag=min_lag_samples,
                max_lag=max_lag_samples,
                dropout=dropout,
                head_dropout=head_dropout
            )
        elif use_sinc or arch == "sinc":
            self.model = SincMultiBandCATCNDecoder(
                eeg_channels=eeg_channels,
                audio_bands=audio_bands,
                hidden_dim=hidden_dim,
                min_lag=min_lag_samples,
                max_lag=max_lag_samples,
                head_type=head_type,
                dropout=dropout,
                head_dropout=head_dropout
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

    @property
    def spatial_head(self):
        if hasattr(self.model, "spatial_head"):
            return self.model.spatial_head
        return None
        
    def forward(self, eeg, audio_a, audio_b, return_spatial: bool = False):
        if hasattr(self.model, "forward"):
            import inspect
            sig = inspect.signature(self.model.forward)
            if "return_spatial" in sig.parameters:
                return self.model(eeg, audio_a, audio_b, return_spatial=return_spatial)
        res = self.model(eeg, audio_a, audio_b)
        if return_spatial:
            s_dir = torch.zeros(eeg.size(0), device=eeg.device)
            return res[0], res[1], res[2], s_dir
        return res

    def predict_spatial_direction(self, eeg):
        if hasattr(self.model, "predict_spatial_direction"):
            return self.model.predict_spatial_direction(eeg)
        return torch.zeros(eeg.size(0), device=eeg.device)

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
