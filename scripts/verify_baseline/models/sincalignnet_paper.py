import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SincConv1d(nn.Module):
    """
    Sinc-based 1D convolution parameterized by cutoff frequencies.
    """
    def __init__(self, out_channels, kernel_size, sample_rate, min_low_hz, min_band_hz, in_channels=1, stride=1, padding=0):
        super(SincConv1d, self).__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            self.kernel_size = kernel_size + 1
            
        self.stride = stride
        self.padding = padding
        self.sample_rate = sample_rate

        # Initialize cutoffs linearly
        hz = np.linspace(min_low_hz, sample_rate / 2, out_channels + 1)
        
        f1 = hz[:-1]
        f2 = hz[1:]
        
        # Enforce minimum band constraints on initialization
        band_hz = f2 - f1
        band_hz = np.maximum(band_hz, min_band_hz)
        f2 = f1 + band_hz
        
        # Normalize to [0, 1] (Nyquist = 0.5)
        f1 = f1 / sample_rate
        f2 = f2 / sample_rate
        
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
        x: [B, 1, C, T] (for EEG) or [B, 1, 1, T] (for Audio)
        """
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
        
        return F.conv2d(x, filters, stride=self.stride, padding=self.padding)

class SincAlignEEGEncoder(nn.Module):
    """
    Faithful Replication of SincAlignNet EEG Encoder.
    """
    def __init__(self, in_channels=8, sample_rate=64, seq_len=640):
        super().__init__()
        
        # 1. Multi-SincNet Bandpass
        self.sinc_conv = SincConv1d(
            out_channels=60, 
            kernel_size=31, 
            sample_rate=sample_rate,
            min_low_hz=1.0,
            min_band_hz=4.0,
            in_channels=1,
            padding=(0, 15) # padding to keep T same
        )
        
        # 2. Depth Conv1D (Assuming depthwise over the 60 filters)
        # SincConv outputs [B, 60, C, T]. 
        # Standard Depth Conv1D in audio takes [B, C_in, T] and outputs [B, C_out, T].
        # For EEG, we flatten C into the channel dimension: 60 * in_channels.
        # We use a standard 1D conv that outputs 32 channels.
        self.depth_conv = nn.Conv1d(
            in_channels=60 * in_channels, 
            out_channels=32, 
            kernel_size=3,
            padding=1
        )
        
        # 3. Downsampling
        self.pool1 = nn.MaxPool1d(kernel_size=6, stride=6)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # Calculate flattened dimension dynamically
        # Conv1d padding=1 keeps T. 
        # Pool1: T // 6
        # Pool2: (T // 6) // 4
        pooled_T = (seq_len // 6) // 4
        flattened_dim = 32 * pooled_T
        
        # 4. Projector (Linear Layers)
        self.projector = nn.Sequential(
            nn.Linear(flattened_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, x):
        # x: [B, C, T]
        B, C, T = x.shape
        x = x.unsqueeze(1) # [B, 1, C, T]
        
        x = self.sinc_conv(x) # [B, 60, C, T]
        
        # Flatten filters and channels into a single dimension for Depth Conv1D
        x = x.view(B, 60 * C, T) 
        
        x = self.depth_conv(x) # [B, 32, T]
        x = F.relu(x)
        
        x = self.pool1(x)
        x = self.pool2(x)
        
        x = x.view(B, -1) # Flatten
        
        x = self.projector(x) # [B, 128]
        return x

class SincAlignAudioEncoder(nn.Module):
    """
    Faithful Replication of SincAlignNet Audio Encoder.
    """
    def __init__(self, sample_rate=16000, seq_len=16000):
        super().__init__()
        
        # 1. Multi-SincNet Bandpass
        self.sinc_conv = SincConv1d(
            out_channels=320, 
            kernel_size=101, 
            sample_rate=sample_rate,
            min_low_hz=50.0,
            min_band_hz=50.0,
            in_channels=1,
            padding=(0, 50) # padding to keep T same
        )
        
        self.depth_conv = nn.Conv1d(
            in_channels=320, 
            out_channels=32, 
            kernel_size=3,
            padding=1
        )
        
        self.pool1 = nn.MaxPool1d(kernel_size=6, stride=6)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        pooled_T = (seq_len // 6) // 4
        flattened_dim = 32 * pooled_T
        
        self.projector = nn.Sequential(
            nn.Linear(flattened_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, x):
        # x: [B, 1, T]
        B, C, T = x.shape
        x = x.unsqueeze(1) # [B, 1, 1, T]
        
        x = self.sinc_conv(x) # [B, 320, 1, T]
        x = x.squeeze(2) # [B, 320, T]
        
        x = self.depth_conv(x) # [B, 32, T]
        x = F.relu(x)
        
        x = self.pool1(x)
        x = self.pool2(x)
        
        x = x.view(B, -1) # Flatten
        
        x = self.projector(x) # [B, 128]
        return x

if __name__ == "__main__":
    # Test EEG Encoder with 1s window (64 samples)
    eeg_net = SincAlignEEGEncoder(in_channels=6, sample_rate=64, seq_len=64)
    dummy_eeg = torch.randn(4, 6, 64)
    out_eeg = eeg_net(dummy_eeg)
    print(f"EEG Output Shape (1s): {out_eeg.shape}")
    eeg_params = sum(p.numel() for p in eeg_net.parameters() if p.requires_grad)
    print(f"EEG Parameters: {eeg_params:,}")
    
    # Test Audio Encoder with 1s window (16000 samples)
    audio_net = SincAlignAudioEncoder(sample_rate=16000, seq_len=16000)
    dummy_audio = torch.randn(4, 1, 16000)
    out_audio = audio_net(dummy_audio)
    print(f"Audio Output Shape (1s): {out_audio.shape}")
    audio_params = sum(p.numel() for p in audio_net.parameters() if p.requires_grad)
    print(f"Audio Parameters: {audio_params:,}")
