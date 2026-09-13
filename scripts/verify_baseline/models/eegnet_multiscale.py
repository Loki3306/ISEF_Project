import torch
import torch.nn as nn

class MultiScaleTemporal(nn.Module):
    def __init__(self, in_channels=1, F1=8, kernels=[3, 7, 15, 31]):
        super().__init__()
        assert F1 % len(kernels) == 0, f"F1 ({F1}) must be divisible by number of kernels ({len(kernels)})"
        filters_per_branch = F1 // len(kernels)
        
        self.branches = nn.ModuleList([
            nn.Conv2d(in_channels, filters_per_branch, (1, k), padding=(0, k//2), bias=False)
            for k in kernels
        ])
        
    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        return torch.cat(outs, dim=1)

class EEGNetMultiScale(nn.Module):
    """
    EEGNet adapted for continuous auditory envelope reconstruction (regression).
    Replaces the single temporal convolution with a multi-scale temporal block (k=3, 7, 15, 31).
    """
    def __init__(self, in_channels=8, F1=8, D=2, F2=16, kernels=[3, 7, 15, 31]):
        super().__init__()
        
        self.temporal_conv = MultiScaleTemporal(in_channels=1, F1=F1, kernels=kernels)
        
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
        # Input: [Batch, Channels, Time]
        x = x.unsqueeze(1) # [Batch, 1, Channels, Time]
        x = self.temporal_conv(x) # [Batch, F1, Channels, Time_padded]
        x = self.spatial_conv(x)  # [Batch, F1*D, 1, Time_padded]
        x = self.block2(x) # [Batch, F2, 1, Time_padded_more]
        x = x.squeeze(2)   # [Batch, F2, Time]
        x = self.output_proj(x) # [Batch, 1, Time]
        return x[..., :orig_len]

def print_summary():
    model = EEGNetMultiScale()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EEGNetMultiScale Parameter Count: {params:,}")
    
if __name__ == "__main__":
    print_summary()
