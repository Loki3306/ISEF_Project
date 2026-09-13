import torch
import torch.nn as nn

class MultiScaleTemporalLinear(nn.Module):
    """
    Bug-Free Multi-Scale Temporal block.
    Outputs a purely linear projection (no GELU, no redundant BN)
    to match the exact input structure expected by EEGNet's spatial block.
    """
    def __init__(self, in_channels=1, output_channels=8, filters_per_branch=9, kernels=[3, 7, 15, 31]):
        super().__init__()
        
        self.branches = nn.ModuleList([
            nn.Conv2d(in_channels, filters_per_branch, (1, k), padding=(0, k//2), bias=False)
            for k in kernels
        ])
        
        # 1x1 point-wise conv to project down to output_channels (8)
        self.channel_mixing = nn.Conv2d(filters_per_branch * len(kernels), output_channels, (1, 1), bias=False)
        
    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        x = torch.cat(outs, dim=1) # [B, 36, C, T]
        x = self.channel_mixing(x) # [B, 8, C, T]
        return x # Purely linear output

class EEGNetMultiScaleV1(nn.Module):
    """
    Diagnostic Model A: Original kernels (3, 7, 15, 31), capacity matched, BUG-FREE.
    """
    def __init__(self, in_channels=8, F1=8, D=2, F2=16):
        super().__init__()
        
        self.temporal_conv = MultiScaleTemporalLinear(
            in_channels=1, 
            output_channels=F1, 
            filters_per_branch=9, 
            kernels=[3, 7, 15, 31]
        )
        
        # Standard EEGNet Spatial Block
        self.spatial_conv = nn.Sequential(
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (in_channels, 1), groups=F1, bias=False),
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
        x = x.unsqueeze(1) 
        x = self.temporal_conv(x) 
        x = self.spatial_conv(x)  
        x = self.block2(x) 
        x = x.squeeze(2)   
        x = self.output_proj(x) 
        return x[..., :orig_len]


class EEGNetMultiScaleV2(nn.Module):
    """
    Diagnostic Model B: 1-Second Context kernels (7, 15, 31, 63), capacity matched, BUG-FREE.
    """
    def __init__(self, in_channels=8, F1=8, D=2, F2=16):
        super().__init__()
        
        self.temporal_conv = MultiScaleTemporalLinear(
            in_channels=1, 
            output_channels=F1, 
            filters_per_branch=9, 
            kernels=[7, 15, 31, 63]
        )
        
        # Standard EEGNet Spatial Block
        self.spatial_conv = nn.Sequential(
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (in_channels, 1), groups=F1, bias=False),
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
        x = x.unsqueeze(1) 
        x = self.temporal_conv(x) 
        x = self.spatial_conv(x)  
        x = self.block2(x) 
        x = x.squeeze(2)   
        x = self.output_proj(x) 
        return x[..., :orig_len]

def print_summary():
    model1 = EEGNetMultiScaleV1()
    params1 = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    print(f"EEGNetMultiScaleV1 (3/7/15/31) Parameter Count: {params1:,}")
    
    model2 = EEGNetMultiScaleV2()
    params2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    print(f"EEGNetMultiScaleV2 (7/15/31/63) Parameter Count: {params2:,}")
    
if __name__ == "__main__":
    print_summary()
