import torch
import torch.nn as nn
from config import isef_config as cfg
from models.msca_modules import (
    MultiScaleTemporalBlock, 
    PhysiologicalSincConv
)
from models.isef_architecture import LightweightConformerBlock, SEChannelAttention

class MSCA_EEGEncoder(nn.Module):
    """
    Wraps the MSCA feature extraction (Multi-scale temporal + SincConv + SE Channel Attention + Spatial Mixer + Conformer)
    and outputs a time-series (B, latent_dim, T) so it plugs perfectly into ContrastiveMatchNet.
    """
    def __init__(self, in_channels=8):
        super().__init__()
        
        # 1. Temporal Branch
        self.temporal_block = MultiScaleTemporalBlock(
            in_channels=in_channels,
            out_channels_per_branch=cfg.TEMPORAL_FILTERS_PER_BRANCH,
            kernels=cfg.TEMPORAL_KERNELS
        )
        
        # 2. Spectral Branch
        self.spectral_block = PhysiologicalSincConv(
            in_channels=in_channels,
            kernel_size=31,
            bands_hz=cfg.SINC_BANDS_HZ,
            sr=cfg.SR
        )
        
        self.concat_channels = cfg.TOTAL_TEMPORAL_CHANNELS + (in_channels * cfg.SINC_CHANNELS)
        
        self.se_attention = SEChannelAttention(self.concat_channels)
        
        # Spatial/Channel Mixer
        self.spatial_mixer = nn.Sequential(
            nn.Conv1d(self.concat_channels, cfg.SPATIAL_MIX_CHANNELS, kernel_size=1),
            nn.BatchNorm1d(cfg.SPATIAL_MIX_CHANNELS),
            nn.ELU()
        )
        
        # We do NOT downsample time (no stride=4) because ContrastiveMatchNet expects matching 
        # time dimensions for EEG and Audio.
        self.proj = nn.Sequential(
            nn.Conv1d(cfg.SPATIAL_MIX_CHANNELS, cfg.CONFORMER_D_MODEL, kernel_size=3, padding=1),
            nn.BatchNorm1d(cfg.CONFORMER_D_MODEL),
            nn.ELU()
        )
        
        # Conformer Blocks
        self.conformer = nn.Sequential(*[
            LightweightConformerBlock(
                d_model=cfg.CONFORMER_D_MODEL, 
                num_heads=cfg.CONFORMER_HEADS, 
                ffn_dim=cfg.CONFORMER_FFN
            ) for _ in range(cfg.CONFORMER_BLOCKS)
        ])
        
        # Placeholder for final projection (overridden by ContrastiveMatchNet)
        self.output_proj = nn.Identity()

    def forward(self, x):
        """ x: (B, C, T) """
        t_feat = self.temporal_block(x) # (B, 64, T)
        s_feat = self.spectral_block(x) # (B, 64, T)
        
        out = torch.cat([t_feat, s_feat], dim=1) # (B, 128, T)
        out = self.se_attention(out)
        out = self.spatial_mixer(out) # (B, 64, T)
        
        out = self.proj(out) # (B, 64, T)
        
        # Conformer expects (B, T, D)
        out = out.transpose(1, 2) # (B, T, 64)
        out = self.conformer(out)
        out = out.transpose(1, 2) # (B, 64, T)
        
        out = self.output_proj(out)
        
        return out
