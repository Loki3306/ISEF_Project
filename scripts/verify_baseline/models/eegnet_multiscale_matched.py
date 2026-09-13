import torch
import torch.nn as nn

class MultiScaleTemporalMatched(nn.Module):
    def __init__(self, in_channels=1, output_channels=8, filters_per_branch=9, kernels=[3, 7, 15, 31]):
        super().__init__()
        
        self.branches = nn.ModuleList([
            nn.Conv2d(in_channels, filters_per_branch, (1, k), padding=(0, k//2), bias=False)
            for k in kernels
        ])
        
        # 1x1 point-wise conv to project down to output_channels (8)
        self.channel_mixing = nn.Conv2d(filters_per_branch * len(kernels), output_channels, (1, 1), bias=False)
        self.mixing_bn = nn.BatchNorm2d(output_channels)
        self.mixing_act = nn.GELU()
        
    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        x = torch.cat(outs, dim=1) # [B, 36, C, T]
        x = self.channel_mixing(x) # [B, 8, C, T]
        x = self.mixing_bn(x)
        x = self.mixing_act(x)
        return x

class EEGNetMultiScaleMatched(nn.Module):
    """
    EEGNet adapted for continuous auditory envelope reconstruction (regression).
    Replaces the single temporal convolution with a capacity-matched multi-scale block.
    """
    def __init__(self, in_channels=8, F1=8, D=2, F2=16, kernels=[3, 7, 15, 31]):
        super().__init__()
        
        self.temporal_conv = MultiScaleTemporalMatched(
            in_channels=1, 
            output_channels=F1, 
            filters_per_branch=9, 
            kernels=kernels
        )
        
        # Spatial conv doesn't need its own BN anymore because mixing_bn handles it,
        # but to keep it strictly identical to standard EEGNet's spatial block input,
        # we can just pass the output directly. Standard EEGNet does BN -> Spatial.
        # So we leave the Spatial block identical.
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
    model = EEGNetMultiScaleMatched()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EEGNetMultiScaleMatched Parameter Count: {params:,}")
    
if __name__ == "__main__":
    print_summary()
