import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from models.eegnet import EEGNet
from models.eegnet_s2 import EEGNetS2

def print_parameter_accounting():
    s0_model = EEGNet()
    s2_model = EEGNetS2()
    
    s0_total = sum(p.numel() for p in s0_model.parameters() if p.requires_grad)
    s2_total = sum(p.numel() for p in s2_model.parameters() if p.requires_grad)
    delta = s2_total - s0_total
    
    print("==================================================")
    print("PARAMETER ACCOUNTING")
    print("==================================================")
    print(f"S0/E1 parameters: {s0_total:,}")
    print(f"S2 parameters:    {s2_total:,}")
    print(f"Additional params: +{delta:,}")
    print("--------------------------------------------------")
    
    # Breakdown of S2 spatial block
    attn_params = sum(p.numel() for p in s2_model.spatial_attention.mha.parameters() if p.requires_grad)
    proj_params = sum(p.numel() for p in s2_model.spatial_attention.weight_proj.parameters() if p.requires_grad)
    other_spatial = sum(p.numel() for p in s2_model.spatial_bn_act.parameters() if p.requires_grad)
    
    # S0 spatial block parameters
    s0_spatial = sum(p.numel() for p in s0_model.block1[2].parameters() if p.requires_grad) # Conv2d
    s0_spatial_bn = sum(p.numel() for p in s0_model.block1[3].parameters() if p.requires_grad) # BatchNorm2d
    
    print("S2 Spatial Breakdown:")
    print(f"  Attention (MHA) params: {attn_params:,}")
    print(f"  Weight projection params: {proj_params:,}")
    print(f"  Spatial BN/Act params: {other_spatial:,}")
    print(f"  Total S2 spatial params: {attn_params + proj_params + other_spatial:,}")
    print("--------------------------------------------------")
    print(f"S0 Spatial Breakdown:")
    print(f"  Depthwise Conv params: {s0_spatial:,}")
    print(f"  Spatial BN/Act params: {s0_spatial_bn:,}")
    print(f"  Total S0 spatial params: {s0_spatial + s0_spatial_bn:,}")
    print("==================================================")

if __name__ == "__main__":
    print_parameter_accounting()
