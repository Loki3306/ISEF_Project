"""
Dual-Stream Cross-Modal Neuro-Conformer (NeuroConformer v4).
Physiologically-Grounded Cross-Modal Attention Architecture for Auditory Attention Decoding.

Architectural Highlights:
1. Bilateral Dipole Spatial Beamformer:
   Combines 8 monopolar wearable EEG electrodes with 4 hemispheric differential dipole pairs:
   - Frontal: Fp1 - Fp2
   - Anterior Temporal: F7 - F8
   - Mid-Temporal (Primary Auditory Cortex / Heschl's Gyrus): T7 - T8
   - Posterior Temporal: P7 - P8
   Outputs 12 spatial channels to isolate auditory cortex lateralization.
2. Parameterized Biological SincNet (SincConvEEG):
   8 learnable auditory cortical tracking bandpass filters (1.0 to 6.5 Hz) covering prosody,
   word stress, syllable rate, and phonemic transitions with Hamming windowing.
   Produces 96 spatial-spectral channels (12 x 8).
3. Conformer Intra-Modal Backbones:
   Macaron-style Conformer blocks (FFN1 half-step -> MHSA -> Depthwise Conv -> FFN2 half-step -> LayerNorm)
   for both EEG and Audio subband representations.
4. Causal Physiological Cross-Modal Attention:
   Allows cortical EEG tokens to directly query cochlear speech tokens constrained to a physiological
   evoked latency window tau in [-2, +18] samples (-31 ms to +281 ms) to model the biological N100/P200 ERP.
5. Siamese Stream Scoring with Exact Anti-Symmetry:
   Score S(EEG, Audio) evaluated via identical shared weights for candidate streams A and B.
   Delta(A, B) = S_A - S_B -> Delta(B, A) = -Delta(A, B) exact to machine precision (0.00e+00 discrepancy).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------------------------------
# 1. BIOLOGICAL SPATIAL & SPECTRAL FRONT-ENDS
# -------------------------------------------------------------------------------------------------

class BilateralDipoleBeamformer(nn.Module):
    """
    Extracts hemispheric differential dipole pairs from wearable EEG channels
    and concatenates them with the original channels.
    - 8 Channels: 4 dipoles (Fp1-Fp2, F7-F8, T7-T8, P7-P8) -> 12 spatial channels.
    - 16 Channels (Dual-Band: 8ch ERP + 8ch Alpha): 4 ERP dipoles + 4 Alpha dipoles -> 24 spatial channels.
    """
    def __init__(self, in_channels: int = 8):
        super().__init__()
        self.in_channels = in_channels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        if x.shape[1] == 8:
            d0 = x[:, 0:1, :] - x[:, 1:2, :]  # L1 - R1 (Frontal / Anterior)
            d1 = x[:, 2:3, :] - x[:, 3:4, :]  # L2 - R2 (Mid-Temporal)
            d2 = x[:, 4:5, :] - x[:, 5:6, :]  # L3 - R3 (Auditory Cortex / Heschl's)
            d3 = x[:, 6:7, :] - x[:, 7:8, :]  # L4 - R4 (Posterior Temporal / Parietal)
            return torch.cat([x, d0, d1, d2, d3], dim=1)  # (B, 12, T)
        elif x.shape[1] == 16:
            # Band 1: ERP (0..7)
            d_erp0 = x[:, 0:1, :] - x[:, 1:2, :]
            d_erp1 = x[:, 2:3, :] - x[:, 3:4, :]
            d_erp2 = x[:, 4:5, :] - x[:, 5:6, :]
            d_erp3 = x[:, 6:7, :] - x[:, 7:8, :]
            # Band 2: Alpha (8..15)
            d_alp0 = x[:, 8:9, :] - x[:, 9:10, :]
            d_alp1 = x[:, 10:11, :] - x[:, 11:12, :]
            d_alp2 = x[:, 12:13, :] - x[:, 13:14, :]
            d_alp3 = x[:, 14:15, :] - x[:, 15:16, :]
            return torch.cat([
                x[:, :8, :], d_erp0, d_erp1, d_erp2, d_erp3,
                x[:, 8:, :], d_alp0, d_alp1, d_alp2, d_alp3
            ], dim=1)  # (B, 24, T)
        return x



class SincConvEEG(nn.Module):
    """
    Parameterized Biological SincNet Filterbank for Scalp EEG.
    Extracts 8 physiological neural oscillation bands in 1.0 - 6.5 Hz.
    """
    def __init__(self, out_bands: int = 8, kernel_size: int = 65, sample_rate: float = 64.0):
        super().__init__()
        self.out_bands = out_bands
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2.0
        
        # 8 biological auditory tracking bands (1.00 to 6.50 Hz)
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
        f2 = torch.clamp(f1 + band, max=self.nyquist - 0.1)
        return f1 / self.sample_rate, f2 / self.sample_rate
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T = x.shape
        f1, f2 = self.get_bands()
        t = self.t_right.unsqueeze(0)  # (1, K_half)
        
        f1_2pi_t = 2.0 * math.pi * f1.unsqueeze(1) * t
        f2_2pi_t = 2.0 * math.pi * f2.unsqueeze(1) * t
        
        h_right = (torch.sin(f2_2pi_t) - torch.sin(f1_2pi_t)) / (math.pi * t)
        h_center = 2.0 * (f2 - f1).unsqueeze(1)
        h_left = torch.flip(h_right, dims=[1])
        
        filters = torch.cat([h_left, h_center, h_right], dim=1) * self.window.unsqueeze(0)  # (out_bands, K)
        filters = filters / (2.0 * (f2 - f1).unsqueeze(1) + 1e-8)
        
        # Depthwise over spatial channels: (B, C, T) -> (B, C * out_bands, T)
        pad = self.kernel_size // 2
        filters_4d = filters.unsqueeze(1).repeat(C, 1, 1)  # (C * out_bands, 1, K)
        x_reshaped = x.view(B * C, 1, T)
        out = F.conv1d(x_reshaped, filters_4d[:self.out_bands], padding=pad)
        out = out.view(B, C, self.out_bands, T).permute(0, 1, 2, 3).contiguous()
        return out.view(B, C * self.out_bands, T)


# -------------------------------------------------------------------------------------------------
# 2. CONFORMER MODULES (MACARON FEEDFORWARD + MHSA + CONV MODULE)
# -------------------------------------------------------------------------------------------------

class FeedForwardModule(nn.Module):
    """
    Conformer Feed-Forward Module with Swish/SiLU and Dropout.
    """
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConformerConvModule(nn.Module):
    """
    Conformer Convolution Module:
    LayerNorm -> Pointwise Conv -> GLU -> Depthwise Conv1d -> BatchNorm -> SiLU -> Pointwise Conv -> Dropout.
    """
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd for same padding"
        self.ln = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1, bias=False)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=d_model, bias=False
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pointwise2 = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        residual = x
        x = self.ln(x).transpose(1, 2)  # (B, D, T)
        x = self.glu(self.pointwise1(x))
        x = self.depthwise(x)
        x = self.act(self.bn(x))
        x = self.dropout(self.pointwise2(x))
        return residual + x.transpose(1, 2)  # (B, T, D)


class DifferentialMultiheadAttention(nn.Module):
    """
    Differential Multi-Head Attention Mechanism (AFA-Net / DiffTransformer).
    Computes DiffAttn = (Softmax(A1) - lambda * Softmax(A2)) * V
    Actively cancels common-mode background EEG neural noise and myogenic artifacts.
    """
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1, lambda_init: float = 0.8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads
        assert self.head_dim % 2 == 0, "head_dim must be even to split into 2 sub-heads"
        self.d_sub = self.head_dim // 2
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Learnable lambda parameters per head for noise cancellation
        self.lambda_q1 = nn.Parameter(torch.zeros(num_heads, self.d_sub))
        self.lambda_k1 = nn.Parameter(torch.zeros(num_heads, self.d_sub))
        self.lambda_q2 = nn.Parameter(torch.zeros(num_heads, self.d_sub))
        self.lambda_k2 = nn.Parameter(torch.zeros(num_heads, self.d_sub))
        self.lambda_init = lambda_init
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        # q_in: (B, T_q, D), k_in: (B, T_k, D), v_in: (B, T_k, D)
        B, T_q, D = q_in.shape
        T_k = k_in.shape[1]
        H, d_sub = self.num_heads, self.d_sub
        
        q = self.q_proj(q_in).view(B, T_q, H, 2, d_sub).permute(3, 0, 2, 1, 4)  # (2, B, H, T_q, d_sub)
        k = self.k_proj(k_in).view(B, T_k, H, 2, d_sub).permute(3, 0, 2, 1, 4)  # (2, B, H, T_k, d_sub)
        v = self.v_proj(v_in).view(B, T_k, H, self.head_dim).transpose(1, 2)    # (B, H, T_k, head_dim)
        
        q1, q2 = q[0], q[1]  # (B, H, T_q, d_sub)
        k1, k2 = k[0], k[1]  # (B, H, T_k, d_sub)
        
        scale = 1.0 / math.sqrt(d_sub)
        attn1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale  # (B, H, T_q, T_k)
        attn2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale  # (B, H, T_q, T_k)
        
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn1 = attn1 + attn_mask
            attn2 = attn2 + attn_mask
            
        p1 = F.softmax(attn1, dim=-1)
        p2 = F.softmax(attn2, dim=-1)
        
        lam = (torch.exp((self.lambda_q1 * self.lambda_k1).sum(dim=-1)) - 
               torch.exp((self.lambda_q2 * self.lambda_k2).sum(dim=-1)) + self.lambda_init).clamp(0.0, 1.0)
        lam = lam.view(1, H, 1, 1)
        
        diff_attn = p1 - lam * p2
        diff_attn = self.dropout(diff_attn)
        
        out = torch.matmul(diff_attn, v)  # (B, H, T_q, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T_q, D)
        return self.out_proj(out)


class ConformerBlock(nn.Module):
    """
    Full Macaron-style Conformer Block.
    x -> 0.5*FFN1 -> MHSA (Differential or Standard) -> ConvModule -> 0.5*FFN2 -> LayerNorm
    """
    def __init__(self, d_model: int, num_heads: int = 4, ffn_dim: int = 192, conv_kernel: int = 31,
                 dropout: float = 0.1, use_diff_attn: bool = False):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, ffn_dim, dropout)
        self.ln_sa = nn.LayerNorm(d_model)
        self.use_diff_attn = use_diff_attn
        if use_diff_attn:
            self.self_attn = DifferentialMultiheadAttention(d_model, num_heads, dropout=dropout)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.sa_dropout = nn.Dropout(dropout)
        self.conv_module = ConformerConvModule(d_model, kernel_size=conv_kernel, dropout=dropout)
        self.ffn2 = FeedForwardModule(d_model, ffn_dim, dropout)
        self.final_ln = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + 0.5 * self.ffn1(x)
        
        # Self Attention (Differential or Standard)
        sa_in = self.ln_sa(x)
        if self.use_diff_attn:
            sa_out = self.self_attn(sa_in, sa_in, sa_in)
        else:
            sa_out, _ = self.self_attn(sa_in, sa_in, sa_in)
        x = x + self.sa_dropout(sa_out)
        
        # Conformer Convolution Module
        x = self.conv_module(x)
        
        # Half-step FFN2
        x = x + 0.5 * self.ffn2(x)
        return self.final_ln(x)


# -------------------------------------------------------------------------------------------------
# 3. CAUSAL PHYSIOLOGICAL CROSS-MODAL ATTENTION
# -------------------------------------------------------------------------------------------------

class CausalPhysiologicalCrossAttention(nn.Module):
    """
    Causal Latency-Constrained Differential Cross-Modal Attention.
    Cortical EEG tokens query cochlear audio tokens with a physiological ERP mask
    restricting attention to tau in [min_lag, max_lag] (-31 ms to +281 ms).
    Uses Differential Attention to cancel common-mode background EEG noise.
    """
    def __init__(self, d_model: int, num_heads: int = 4, ffn_dim: int = 192, min_lag: int = -2, max_lag: int = 18, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.min_lag = min_lag
        self.max_lag = max_lag
        
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.cross_attn = DifferentialMultiheadAttention(d_model, num_heads, dropout=dropout)
        self.attn_dropout = nn.Dropout(dropout)
        
        self.ffn = FeedForwardModule(d_model, ffn_dim, dropout)
        self.final_ln = nn.LayerNorm(d_model)
        self._mask_cache: dict[tuple[int, str], torch.Tensor] = {}
        
    def get_latency_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Retrieves or creates a cached additive attention mask of shape (T, T).
        Position (t_eeg, t_audio) is 0.0 if t_eeg - t_audio in [min_lag, max_lag], else -inf.
        Cached by (T, device) to avoid GPU allocation/synchronization stalls in the hot loop.
        """
        key = (T, str(device))
        if key not in self._mask_cache or self._mask_cache[key].device != device:
            t_q = torch.arange(T, device=device).unsqueeze(1)  # (T, 1)
            t_k = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
            lag = t_q - t_k  # lag = t_eeg - t_audio
            valid = (lag >= self.min_lag) & (lag <= self.max_lag)
            mask = torch.full((T, T), float('-inf'), device=device)
            mask[valid] = 0.0
            mask[0, 0] = 0.0
            self._mask_cache[key] = mask
        return self._mask_cache[key]
        
    def forward(self, z_eeg: torch.Tensor, z_audio: torch.Tensor) -> torch.Tensor:
        # z_eeg: (B, T, D), z_audio: (B, T, D)
        B, T, D = z_eeg.shape
        attn_mask = self.get_latency_mask(T, z_eeg.device)
        
        q = self.ln_q(z_eeg)
        kv = self.ln_kv(z_audio)
        
        cross_out = self.cross_attn(q, kv, kv, attn_mask=attn_mask)
        x = z_eeg + self.attn_dropout(cross_out)
        
        x = x + self.ffn(x)
        return self.final_ln(x)  # (B, T, D)


# -------------------------------------------------------------------------------------------------
# 4. TEMPORAL SALIENCE & BILINEAR ALIGNMENT SCORING HEAD WITH RESIDUAL ANCHOR
# -------------------------------------------------------------------------------------------------

class SiameseCrossModalScoringHead(nn.Module):
    """
    Energy-Aware Temporal Salience Weighted Multi-Lag Alignment Head with Residual Linear Anchor.
    Scores the compatibility between cross-attended EEG and Audio stream.
    Guaranteed machine-precision anti-symmetry: Delta(A, B) = Score(EEG, A) - Score(EEG, B).
    """
    def __init__(self, d_model: int = 80, min_lag: int = -2, max_lag: int = 18, dropout: float = 0.2):
        super().__init__()
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.num_lags = max_lag - min_lag + 1
        
        # Temporal Salience Network (Speech onsets vs pauses)
        self.salience_net = nn.Sequential(
            nn.Conv1d(d_model, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(32, 1, kernel_size=1)
        )
        
        # Dual-path bilinear classifier: salience-weighted cross-correlations + unweighted residual anchor
        self.classifier = nn.Linear(d_model * self.num_lags * 2, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def score_stream(self, z_cross: torch.Tensor, z_audio: torch.Tensor) -> torch.Tensor:
        # z_cross: (B, D, T), z_audio: (B, D, T)
        B, D, T = z_cross.shape
        
        # Zero-mean unit-variance temporal normalization
        zc_norm = (z_cross - z_cross.mean(dim=-1, keepdim=True)) / (z_cross.std(dim=-1, keepdim=True) + 1e-8)
        za_norm = (z_audio - z_audio.mean(dim=-1, keepdim=True)) / (z_audio.std(dim=-1, keepdim=True) + 1e-8)
        
        # Dynamic temporal salience weights over audio
        salience_logits = self.salience_net(za_norm)  # (B, 1, T)
        weights = F.softmax(salience_logits, dim=-1)  # (B, 1, T)
        
        corrs = []
        corrs_unw = []
        for tau in range(self.min_lag, self.max_lag + 1):
            if tau > 0:
                zc_s = zc_norm[:, :, tau:]
                za_s = za_norm[:, :, :-tau]
                w_s = weights[:, :, :-tau]
            elif tau < 0:
                zc_s = zc_norm[:, :, :tau]
                za_s = za_norm[:, :, -tau:]
                w_s = weights[:, :, -tau:]
            else:
                zc_s = zc_norm
                za_s = za_norm
                w_s = weights
                
            w_norm = w_s / (w_s.sum(dim=-1, keepdim=True) + 1e-8)
            r_tau = (zc_s * za_s * w_norm).sum(dim=-1)   # Salience-weighted (B, D)
            r_unw = (zc_s * za_s).mean(dim=-1)           # Unweighted residual anchor (B, D)
            corrs.append(r_tau)
            corrs_unw.append(r_unw)
            
        r_all = torch.cat([
            torch.stack(corrs, dim=-1).view(B, -1),
            torch.stack(corrs_unw, dim=-1).view(B, -1)
        ], dim=-1)
        r_all = self.dropout(r_all)
        return self.classifier(r_all).squeeze(-1)  # (B,)
        
    def forward(self, z_cross_a: torch.Tensor, z_audio_a: torch.Tensor,
                z_cross_b: torch.Tensor, z_audio_b: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        sa = self.score_stream(z_cross_a, z_audio_a)
        sb = self.score_stream(z_cross_b, z_audio_b)
        delta = sa - sb
        return delta, (sa, sb)


# -------------------------------------------------------------------------------------------------
# 5. MODULAR ENCODERS FOR FULL DOWNSTREAM PIPELINE COMPATIBILITY
# -------------------------------------------------------------------------------------------------

class NeuroConformer_EEGEncoder(nn.Module):
    """
    Modular EEG Encoder wrapping InstanceNorm, Dipole Beamformer, SincNet, Subsampling, and Diff-Conformer.
    Accepts (B, 8, T) -> returns (B, D, T // subsample_stride).
    """
    def __init__(self, in_channels: int = 8, d_model: int = 80, conformer_blocks: int = 2,
                 num_heads: int = 4, ffn_dim: int = 160, dropout: float = 0.15, subsample_stride: int = 2,
                 use_diff_attn: bool = True):
        super().__init__()
        self.subsample_stride = subsample_stride
        self.beamformer = BilateralDipoleBeamformer(in_channels=in_channels)
        spatial_channels = 24 if in_channels == 16 else (12 if in_channels == 8 else in_channels)
        self.sinc_net = SincConvEEG(out_bands=8, kernel_size=65, sample_rate=64.0)
        self.proj = nn.Sequential(
            nn.Conv1d(spatial_channels * 8, d_model, kernel_size=3, stride=subsample_stride, padding=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.SiLU()
        )
        self.conformer = nn.Sequential(*[
            ConformerBlock(d_model=d_model, num_heads=num_heads, ffn_dim=ffn_dim, conv_kernel=31,
                           dropout=dropout, use_diff_attn=use_diff_attn)
            for _ in range(conformer_blocks)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        # Subject-invariant instance normalization per trial across time
        x_norm = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
        spatial = self.beamformer(x_norm)
        spectral = self.sinc_net(spatial)
        feat = self.proj(spectral)       # (B, D, T // stride)
        feat_t = feat.transpose(1, 2)    # (B, T // stride, D)
        conf_out = self.conformer(feat_t)  # (B, T // stride, D)
        return conf_out.transpose(1, 2)  # (B, D, T // stride)


class NeuroConformer_AudioEncoder(nn.Module):
    """
    Modular Audio Encoder wrapping Tonotopic SE, Subsampled Conv Projection, and Audio Conformer.
    Accepts (B, K, T) -> returns (B, D, T // subsample_stride).
    """
    def __init__(self, in_channels: int = 9, d_model: int = 80, conformer_blocks: int = 2,
                 num_heads: int = 4, ffn_dim: int = 160, dropout: float = 0.15, subsample_stride: int = 2):
        super().__init__()
        self.subsample_stride = subsample_stride
        bottleneck = max(4, in_channels // 2)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, in_channels),
            nn.Sigmoid()
        )
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=15, stride=subsample_stride, padding=7, bias=False),
            nn.BatchNorm1d(d_model),
            nn.SiLU()
        )
        self.conformer = nn.Sequential(*[
            ConformerBlock(d_model=d_model, num_heads=num_heads, ffn_dim=ffn_dim, conv_kernel=31, dropout=dropout)
            for _ in range(conformer_blocks)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, K, T)
        w = self.se(x).unsqueeze(-1)
        feat = self.proj(x * w)          # (B, D, T // stride)
        feat_t = feat.transpose(1, 2)    # (B, T // stride, D)
        conf_out = self.conformer(feat_t)  # (B, T // stride, D)
        return conf_out.transpose(1, 2)  # (B, D, T // stride)


class NeuroConformer_ClassifierHead(nn.Module):
    """
    Modular Classifier Head wrapping Causal Cross-Modal Attention and Siamese Scoring.
    Accepts (z_eeg, z_a, z_b) each of shape (B, D, T // stride) -> returns delta, (la, lb).
    """
    def __init__(self, d_model: int = 80, num_heads: int = 4, ffn_dim: int = 160,
                 min_lag: int = -2, max_lag: int = 18, dropout: float = 0.15, head_dropout: float = 0.25,
                 subsample_stride: int = 2):
        super().__init__()
        # Scale lags to match subsampled temporal resolution
        eff_min_lag = int(math.floor(min_lag / subsample_stride))
        eff_max_lag = int(math.ceil(max_lag / subsample_stride))
        self.cross_attention = CausalPhysiologicalCrossAttention(
            d_model=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            min_lag=eff_min_lag,
            max_lag=eff_max_lag,
            dropout=dropout
        )
        self.scoring_head = SiameseCrossModalScoringHead(
            d_model=d_model,
            min_lag=eff_min_lag,
            max_lag=eff_max_lag,
            dropout=head_dropout
        )
        
    def forward(self, z_eeg: torch.Tensor, z_a: torch.Tensor, z_b: torch.Tensor):
        # z_eeg, z_a, z_b: (B, D, T')
        h_eeg = z_eeg.transpose(1, 2)
        h_a = z_a.transpose(1, 2)
        h_b = z_b.transpose(1, 2)
        
        z_cross_a = self.cross_attention(h_eeg, h_a).transpose(1, 2)  # (B, D, T')
        z_cross_b = self.cross_attention(h_eeg, h_b).transpose(1, 2)  # (B, D, T')
        
        delta, (la, lb) = self.scoring_head(z_cross_a, z_a, z_cross_b, z_b)
        return delta, (la, lb)


# -------------------------------------------------------------------------------------------------
# 6. UNIFIED DUAL-STREAM CROSS-MODAL NEURO-CONFORMER DECODER
# -------------------------------------------------------------------------------------------------

class NeuroConformerDecoder(nn.Module):
    """
    Dual-Stream Cross-Modal Neuro-Conformer (v5).
    Full self-attention Conformer backbones + causal cross-modal attention + exact anti-symmetry.
    Features temporal subsampling stride (default 2) for 4x faster training and enhanced cortical SNR.
    Target parameter count: ~480,000 parameters (with d_model=80).
    """
    def __init__(
        self,
        eeg_channels: int = 8,
        audio_bands: int = 9,
        d_model: int = 80,
        conformer_blocks: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 160,
        min_lag: int = -2,
        max_lag: int = 18,
        dropout: float = 0.15,
        head_dropout: float = 0.25,
        subsample_stride: int = 2
    ):
        super().__init__()
        self.d_model = d_model
        self.subsample_stride = subsample_stride
        self.eeg_encoder = NeuroConformer_EEGEncoder(
            in_channels=eeg_channels,
            d_model=d_model,
            conformer_blocks=conformer_blocks,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            subsample_stride=subsample_stride
        )
        self.audio_encoder = NeuroConformer_AudioEncoder(
            in_channels=audio_bands,
            d_model=d_model,
            conformer_blocks=conformer_blocks,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            subsample_stride=subsample_stride
        )
        self.classifier_head = NeuroConformer_ClassifierHead(
            d_model=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            min_lag=min_lag,
            max_lag=max_lag,
            dropout=dropout,
            head_dropout=head_dropout,
            subsample_stride=subsample_stride
        )
        
    def forward(self, eeg: torch.Tensor, audio_a: torch.Tensor, audio_b: torch.Tensor):
        ze = self.eeg_encoder(eeg)
        za = self.audio_encoder(audio_a)
        zb = self.audio_encoder(audio_b)
        delta, (logit_a, logit_b) = self.classifier_head(ze, za, zb)
        return delta, (logit_a, logit_b), (ze, za, zb)


if __name__ == "__main__":
    model = NeuroConformerDecoder(eeg_channels=8, audio_bands=9, d_model=80, conformer_blocks=2)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Dual-Stream Cross-Modal Neuro-Conformer Total Parameters: {n_params:,}")
    
    model.eval()
    dummy_eeg = torch.randn(2, 8, 320)
    dummy_a = torch.randn(2, 9, 320)
    dummy_b = torch.randn(2, 9, 320)
    
    delta, (la, lb), (ze, za, zb) = model(dummy_eeg, dummy_a, dummy_b)
    print(f"Forward Output Shapes: delta={delta.shape}, la={la.shape}, ze={ze.shape}, za={za.shape}")
    
    # Modular sub-calls (for streaming context)
    ze_sub = model.eeg_encoder(dummy_eeg)
    za_sub = model.audio_encoder(dummy_a)
    zb_sub = model.audio_encoder(dummy_b)
    delta_sub, (la_sub, lb_sub) = model.classifier_head(ze_sub, za_sub, zb_sub)
    assert torch.allclose(delta, delta_sub), "Modular pipeline consistency violation!"
    
    # Machine-Precision Anti-Symmetry Verification
    delta_rev, (lb_rev, la_rev), _ = model(dummy_eeg, dummy_b, dummy_a)
    disc = torch.max(torch.abs(delta + delta_rev)).item()
    print(f"Machine-Precision Anti-Symmetry Error |Delta(A,B) + Delta(B,A)|: {disc:.2e}")
    assert disc < 1e-6, "Anti-symmetry violation detected!"
    print("[PASS]: Perfect anti-symmetry verified!")
