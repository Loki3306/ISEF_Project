import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SincConv_Fast(nn.Module):
    """
    Sinc-based convolution tailored for EEG (1D temporal convolution parameterized by cutoff frequencies).
    """
    def __init__(self, out_channels, kernel_size, sample_rate=64, in_channels=1, stride=1, padding=0):
        super(SincConv_Fast, self).__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            self.kernel_size = kernel_size + 1
            
        self.stride = stride
        self.padding = padding
        self.sample_rate = sample_rate

        # Initialize cutoffs linearly across the physiological band (0.5Hz to Nyquist)
        hz = np.linspace(0.5, sample_rate / 2, out_channels + 1)
        
        f1 = hz[:-1]
        f2 = hz[1:]
        
        # Normalize to [0, 1] (Nyquist = 0.5)
        f1 = f1 / sample_rate
        f2 = f2 / sample_rate
        
        # Parameters for lower cutoff (f1) and bandwidth (band)
        self.f1 = nn.Parameter(torch.Tensor(f1))
        self.band = nn.Parameter(torch.Tensor(f2 - f1))

        # Create window function (Hamming window)
        n = torch.linspace(0, self.kernel_size - 1, self.kernel_size)
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * n / (self.kernel_size - 1))
        self.register_buffer('window', window)

        # Create time axis
        t_right = torch.linspace(1, (self.kernel_size - 1) / 2, steps=int((self.kernel_size - 1) / 2))
        self.register_buffer('t_right', t_right)

    def forward(self, x):
        """
        x: [B, 1, C, T]
        """
        # Ensure f1 > 0 and band > 0
        f1 = torch.abs(self.f1)
        band = torch.abs(self.band)
        f2 = f1 + band
        
        # Sinc function: sin(2*pi*f*t) / (pi*t)
        t = self.t_right
        
        f1_2pi_t = 2 * math.pi * f1.view(-1, 1) * t.view(1, -1)
        f2_2pi_t = 2 * math.pi * f2.view(-1, 1) * t.view(1, -1)
        
        low_pass1 = torch.sin(f1_2pi_t) / (math.pi * t.view(1, -1))
        low_pass2 = torch.sin(f2_2pi_t) / (math.pi * t.view(1, -1))
        
        band_pass_right = low_pass2 - low_pass1
        
        # Center of filter (t=0): 2 * (f2 - f1)
        band_pass_center = 2 * band.view(-1, 1)
        
        # Combine left, center, right
        band_pass_left = torch.flip(band_pass_right, dims=[1])
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        
        # Apply window
        band_pass = band_pass * self.window.view(1, -1)
        
        # Reshape to [out_channels, 1, 1, kernel_size] for Conv2d
        filters = band_pass.view(self.out_channels, 1, 1, self.kernel_size)
        
        # Apply convolution
        return F.conv2d(x, filters, stride=self.stride, padding=self.padding)

class SincAlignNet(nn.Module):
    """
    Faithful Replication of SincAlignNet EEG Encoder.
    Uses SincConv -> Depthwise Spatial -> Temporal -> Downsampling -> Interpolation.
    """
    def __init__(self, in_channels=8, F1=8, D=2, F2=16):
        super().__init__()
        
        # 1. SincNet Frequency Filters
        self.sinc_conv = SincConv_Fast(
            out_channels=F1, 
            kernel_size=65, 
            sample_rate=64, 
            in_channels=1,
            padding=(0, 32)
        )
        
        # 2. Depthwise Spatial Convolution
        # SincAlignNet must mix channels spatially to find the auditory dipole.
        self.spatial_conv = nn.Sequential(
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (in_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.GELU()
        )
        
        # 3. Depthwise Temporal Convolution & Downsampling
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.GELU(),
            nn.AvgPool2d((1, 2)), # Downsampling factor of 2
            nn.Dropout(0.25)
        )
        
        # 4. Projection
        self.output_proj = nn.Conv1d(F2, 16, kernel_size=1)

    def forward(self, x):
        orig_len = x.shape[-1]
        x = x.unsqueeze(1) # [B, 1, C, T]
        
        x = self.sinc_conv(x) # [B, F1, C, T]
        x = self.spatial_conv(x) # [B, F1*D, 1, T]
        x = self.temporal_conv(x) # [B, F2, 1, T/2]
        
        x = x.squeeze(2) # [B, F2, T/2]
        x = self.output_proj(x) # [B, 16, T/2]
        
        # Interpolate back to original length to match audio envelope dimension
        if x.shape[-1] != orig_len:
            x = F.interpolate(x, size=orig_len, mode='linear', align_corners=False)
            
        return x

def print_summary():
    model = SincAlignNet()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SincAlignNet Parameter Count (EEG Encoder): {params:,}")
    
if __name__ == "__main__":
    print_summary()
