import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SincConv1dBiologic(nn.Module):
    """
    Sinc-based 1D convolution with rigorous biological band initialization
    and mathematical bounds preventing Nyquist violation.
    """
    def __init__(self, sample_rate, kernel_size=31, in_channels=1, stride=1, padding=(0, 15)):
        super().__init__()

        self.out_channels = 8 # 8 Biological bands
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        self.stride = stride
        self.padding = padding
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2.0
        
        # Biological Passbands (Hz):
        # 1. Low-Delta: 0.5-2.0
        # 2. High-Delta: 2.0-4.0
        # 3. Theta-1: 4.0-6.0
        # 4. Theta-2: 6.0-8.0
        # 5. Alpha-1: 8.0-10.5
        # 6. Alpha-2: 10.5-13.0
        # 7. Low-Beta: 13.0-20.0
        # 8. Mid-Beta: 20.0-30.0
        f1_init_hz = torch.tensor([0.5, 2.0, 4.0, 6.0, 8.0, 10.5, 13.0, 20.0])
        f2_init_hz = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.5, 13.0, 20.0, 30.0])
        band_init_hz = f2_init_hz - f1_init_hz
        
        self.min_low_hz = 0.1
        self.min_band_hz = 1.0

        # Initialize f1 bounds
        f1_norm = (f1_init_hz - self.min_low_hz) / (self.nyquist - self.min_low_hz - self.min_band_hz + 1e-6)
        f1_norm = torch.clamp(f1_norm, 1e-4, 1.0 - 1e-4)
        f1_raw_init = torch.log(f1_norm / (1 - f1_norm))
        
        # Initialize band bounds
        band_norm = (band_init_hz - self.min_band_hz) / (self.nyquist - f1_init_hz - self.min_band_hz + 1e-6)
        band_norm = torch.clamp(band_norm, 1e-4, 1.0 - 1e-4)
        band_raw_init = torch.log(band_norm / (1 - band_norm))
        
        self.f1_raw = nn.Parameter(f1_raw_init)
        self.band_raw = nn.Parameter(band_raw_init)

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
        
        f2 = f1 + band
        
        f1_norm = f1 / self.sample_rate
        f2_norm = f2 / self.sample_rate
        band_norm = band / self.sample_rate
        
        return f1_norm, f2_norm, band_norm

    def forward(self, x):
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
        filters = band_pass.view(self.out_channels, 1, 1, self.kernel_size)
        
        return F.conv2d(x, filters, stride=self.stride, padding=self.padding)


class MultiScaleTemporal(nn.Module):
    def __init__(self, in_channels=1, base_filters=4):
        super().__init__()
        # We process each electrode (in_channels=1) temporally across 4 scales
        self.conv3 = nn.Conv2d(1, base_filters, (1, 3), padding=(0, 1), bias=False)
        self.conv7 = nn.Conv2d(1, base_filters, (1, 7), padding=(0, 3), bias=False)
        self.conv15 = nn.Conv2d(1, base_filters, (1, 15), padding=(0, 7), bias=False)
        self.conv31 = nn.Conv2d(1, base_filters, (1, 31), padding=(0, 15), bias=False)
        self.out_channels = base_filters * 4
        
    def forward(self, x):
        # x: [B, 1, C, T]
        c3 = self.conv3(x)
        c7 = self.conv7(x)
        c15 = self.conv15(x)
        c31 = self.conv31(x)
        return torch.cat([c3, c7, c15, c31], dim=1) # [B, out_channels, C, T]


class SEAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        
    def forward(self, x):
        # x: [B, C, T]
        b, c, t = x.size()
        y = x.mean(dim=2) # Global average pool [B, C]
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1)
        return x * y


class MSCA_EEGEncoder(nn.Module):
    def __init__(self, in_channels=8, sample_rate=64, D=2, conformer_dim=64):
        super().__init__()
        
        # 1. Feature Branches
        self.temporal_branch = MultiScaleTemporal(in_channels=1, base_filters=4) # 16 features
        self.spectral_branch = SincConv1dBiologic(sample_rate=sample_rate, kernel_size=31) # 8 features
        
        # Total concatenated features = 16 + 8 = 24
        F1 = 24
        
        # Normalize the raw EEG before spatial mixing
        self.bn_temp = nn.BatchNorm2d(16)
        self.bn_spec = nn.BatchNorm2d(8)
        
        # 2. True Spatial Topology (Depthwise)
        # We learn D spatial filters for each of the 24 temporal/spectral features.
        self.spatial_conv = nn.Conv2d(F1, F1 * D, (in_channels, 1), groups=F1, bias=False)
        self.bn_spatial = nn.BatchNorm2d(F1 * D)
        self.dropout1 = nn.Dropout(0.25)
        
        # 3. SE Channel Attention
        self.se = SEAttention(channels=F1 * D, reduction=4)
        
        # 4. Pointwise Conv to project to conformer_dim
        self.pointwise = nn.Conv1d(F1 * D, conformer_dim, kernel_size=1, bias=False)
        self.bn_point = nn.BatchNorm1d(conformer_dim)
        self.dropout2 = nn.Dropout(0.25)
        
        # 5. Lightweight Conformer (TransformerEncoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=conformer_dim, 
            nhead=4, 
            dim_feedforward=128, 
            dropout=0.2, 
            batch_first=True
        )
        self.conformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 6. Final Sequence Projection
        self.output_proj = nn.Conv1d(conformer_dim, 16, kernel_size=1) # Standard EEGNet F2 output dimension

    def forward(self, x):
        # x: [B, C, T]
        orig_len = x.shape[-1]
        x = x.unsqueeze(1) # [B, 1, C, T]
        
        # Parallel Branches
        t_feat = self.temporal_branch(x) # [B, 16, C, T]
        t_feat = self.bn_temp(t_feat)
        
        # SincConv expects [B, 1, C, T]
        B, _, C, T = x.shape
        s_feat = self.spectral_branch(x) # [B, 8, C, T]
        s_feat = self.bn_spec(s_feat)
        
        # Concatenate Features
        feat = torch.cat([t_feat, s_feat], dim=1) # [B, 24, C, T]
        
        # Spatial Mixing
        feat = self.spatial_conv(feat) # [B, 48, 1, T]
        feat = self.bn_spatial(feat)
        feat = F.gelu(feat)
        feat = self.dropout1(feat)
        
        feat = feat.squeeze(2) # [B, 48, T]
        
        # SE Attention
        feat = self.se(feat)
        
        # Project to Conformer Dim
        feat = self.pointwise(feat) # [B, 64, T]
        feat = self.bn_point(feat)
        feat = F.gelu(feat)
        feat = self.dropout2(feat)
        
        # Conformer expects [B, T, D] if batch_first=True
        feat = feat.transpose(1, 2) # [B, T, 64]
        feat = self.conformer(feat) # [B, T, 64]
        feat = feat.transpose(1, 2) # [B, 64, T]
        
        feat = self.output_proj(feat) # [B, 16, T]
        return feat[..., :orig_len]

if __name__ == "__main__":
    model = MSCA_EEGEncoder()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MSCA Parameter Count: {params:,}")
    dummy = torch.randn(4, 8, 640) # 10 seconds at 64Hz
    out = model(dummy)
    print(f"Output Shape: {out.shape}")
