import torch
import torch.nn as nn

class SpatialMLP(nn.Module):
    def __init__(self, channels=8, F1=8, hidden_dim=16, D=2):
        super().__init__()
        # MLP across the electrode dimension (channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, D)
        )
        
    def forward(self, x):
        # x: [B, F1, channels, T]
        x = x.transpose(-1, -2) # [B, F1, T, channels]
        x = self.mlp(x)         # [B, F1, T, D]
        x = x.transpose(-1, -2) # [B, F1, D, T]
        
        # Reshape to match downstream EEGNet expectations: [B, F1*D, 1, T]
        B, F1, D, T = x.shape
        x = x.reshape(B, F1 * D, 1, T)
        return x

class EEGNetS1(nn.Module):
    """
    Block 4 - S1: Static Nonlinear Cross-Electrode Mixing
    Replaces the linear depthwise spatial convolution of EEGNet with a Spatial MLP
    that acts explicitly on the 8 physical electrodes, without mixing the F1 temporal feature dimension.
    """
    def __init__(self, in_channels=8, F1=8, D=2, F2=16, kernel_length=64, hidden_dim=16):
        super().__init__()
        
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length//2), bias=False),
            nn.BatchNorm2d(F1)
        )
        
        self.spatial_mlp = SpatialMLP(channels=in_channels, F1=F1, hidden_dim=hidden_dim, D=D)
        
        self.spatial_bn_act = nn.Sequential(
            nn.BatchNorm2d(F1 * D),
            nn.GELU(),
            nn.Dropout(0.25)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.GELU(),
            nn.Dropout(0.25)
        )
        
        self.output_proj = nn.Conv1d(F2, 1, kernel_size=1)

    def forward(self, x):
        orig_len = x.shape[-1]
        # Input: [Batch, Channels, Time]
        x = x.unsqueeze(1) # [Batch, 1, Channels, Time]
        
        x = self.temporal_conv(x) # [Batch, F1, Channels, Time+1]
        x = self.spatial_mlp(x)   # [Batch, F1*D, 1, Time+1]
        x = self.spatial_bn_act(x)# [Batch, F1*D, 1, Time+1]
        
        x = self.block2(x) # [Batch, F2, 1, Time+2]
        x = x.squeeze(2)   # [Batch, F2, Time+2]
        x = self.output_proj(x) # [Batch, 1, Time+2]
        return x[..., :orig_len]

def print_summary():
    model = EEGNetS1()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    spatial_params = sum(p.numel() for p in model.spatial_mlp.parameters() if p.requires_grad)
    print(f"EEGNetS1 Total Parameter Count: {params:,}")
    print(f"EEGNetS1 Spatial Parameter Count: {spatial_params:,}")
    
if __name__ == "__main__":
    print_summary()
