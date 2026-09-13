import torch
import torch.nn as nn

class EEGNetMultiScaleM2(nn.Module):
    """
    Diagnostic Model M2: Parameter-Controlled Multi-Scale
    - Allocates 8 total filters across k={15, 31, 63, 127}
    - Total parameters: 520 (approx matched to baseline 512)
    - No 36->8 projection bottleneck
    - Outputs exactly 8 temporal channels for spatial filtering
    """
    def __init__(self, in_channels=8, F1=8, D=2, F2=16):
        super().__init__()
        
        assert F1 == 8, "M2 is hardcoded for exactly 8 output filters"
        
        # Allocations: k=15 (1 filter), k=31 (2 filters), k=63 (3 filters), k=127 (2 filters)
        self.branch_15 = nn.Conv2d(1, 1, (1, 15), padding=(0, 15//2), bias=False)
        self.branch_31 = nn.Conv2d(1, 2, (1, 31), padding=(0, 31//2), bias=False)
        self.branch_63 = nn.Conv2d(1, 3, (1, 63), padding=(0, 63//2), bias=False)
        self.branch_127 = nn.Conv2d(1, 2, (1, 127), padding=(0, 127//2), bias=False)
        
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
        
        # Purely linear multi-scale concatenation (No bottleneck, no GELU)
        temporal_out = torch.cat([
            self.branch_15(x),
            self.branch_31(x),
            self.branch_63(x),
            self.branch_127(x)
        ], dim=1) # [B, 8, C, T]
        
        x = self.spatial_conv(temporal_out)  
        x = self.block2(x) 
        x = x.squeeze(2)   
        x = self.output_proj(x) 
        return x[..., :orig_len]

def print_summary():
    model = EEGNetMultiScaleM2()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EEGNetMultiScaleM2 Parameter Count: {params:,}")
    
if __name__ == "__main__":
    print_summary()
