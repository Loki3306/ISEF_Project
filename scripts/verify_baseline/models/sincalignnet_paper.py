import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SincConv1d(nn.Module):
    """
    Sinc-based 1D convolution parameterized by cutoff frequencies, 
    with rigorous mathematical bounds preventing Nyquist violation.
    """
    def __init__(self, out_channels, kernel_size, sample_rate, min_low_hz, min_band_hz, in_channels=1, stride=1, padding=0):
        super(SincConv1d, self).__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        if kernel_size % 2 == 0:
            self.kernel_size = kernel_size + 1
            
        self.stride = stride
        self.padding = padding
        self.sample_rate = sample_rate
        
        self.nyquist = sample_rate / 2.0
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        hz = torch.linspace(min_low_hz, self.nyquist - min_band_hz, out_channels + 1)
        f1_init = hz[:-1]
        
        f1_norm = (f1_init - self.min_low_hz) / (self.nyquist - self.min_low_hz - self.min_band_hz + 1e-6)
        f1_norm = torch.clamp(f1_norm, 1e-4, 1.0 - 1e-4)
        f1_raw_init = torch.log(f1_norm / (1 - f1_norm))
        
        band_init = min_band_hz + 1.0
        band_norm = (band_init - self.min_band_hz) / (self.nyquist - f1_init - self.min_band_hz + 1e-6)
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
        
        # Normalize for math (Hz -> [0, 0.5])
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
        
        # 4. Projector (Pointwise Conv to preserve sequence)
        self.projector = nn.Conv1d(32, 64, kernel_size=1)

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
        
        x = self.projector(x) # [B, 64, T']
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
        
        self.projector = nn.Conv1d(32, 64, kernel_size=1)

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
        
        x = self.projector(x) # [B, 64, T']
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
