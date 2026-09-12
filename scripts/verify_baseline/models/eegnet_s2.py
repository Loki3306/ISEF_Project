import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicElectrodeAttention(nn.Module):
    def __init__(self, channels=8, F1=8, D=2, num_heads=2):
        super().__init__()
        # Self-attention treating electrodes as the sequence dimension
        # embed_dim corresponds to the F1 temporal filters
        self.mha = nn.MultiheadAttention(embed_dim=F1, num_heads=num_heads, batch_first=True)
        
        # Projection from the attended electrode representation to D spatial filters
        self.weight_proj = nn.Linear(F1, D)
        
    def forward(self, x):
        # x: [B, F1, C, T]
        B, F1, C, T = x.shape
        
        # 1. Global Average Pooling over Time to create electrode embeddings
        # x_pool: [B, F1, C]
        x_pool = x.mean(dim=-1)
        
        # Transpose for batch_first=True MHA: [B, C, F1]
        x_seq = x_pool.transpose(1, 2)
        
        # 2. Self-Attention across electrodes
        # attn_out: [B, C, F1]
        attn_out, _ = self.mha(x_seq, x_seq, x_seq)
        
        # 3. Generate dynamic weights per electrode for D spatial filters
        # weights: [B, C, D]
        weights = self.weight_proj(attn_out)
        
        # Softmax over channels (C) to ensure valid spatial filtering map
        weights = F.softmax(weights, dim=1)
        
        # 4. Mix original temporal features based on dynamic spatial weights
        # x: [B, F1, C, T]
        # weights: [B, C, D]
        # Output: [B, F1, D, T]
        x_mixed = torch.einsum('bfct, bcd -> bfdt', x, weights)
        
        # Reshape to match downstream EEGNet expectations: [B, F1*D, 1, T]
        B, F1, D, T = x_mixed.shape
        x_out = x_mixed.reshape(B, F1 * D, 1, T)
        
        return x_out

class EEGNetS2(nn.Module):
    """
    Block 4 - S2: Dynamic Electrode Attention
    Replaces static linear spatial filter with input-dependent electrode attention.
    """
    def __init__(self, in_channels=8, F1=8, D=2, F2=16, kernel_length=64, num_heads=2):
        super().__init__()
        
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length//2), bias=False),
            nn.BatchNorm2d(F1)
        )
        
        self.spatial_attention = DynamicElectrodeAttention(channels=in_channels, F1=F1, D=D, num_heads=num_heads)
        
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
        
        x = self.temporal_conv(x)      # [Batch, F1, Channels, Time+padded]
        x = self.spatial_attention(x)  # [Batch, F1*D, 1, Time+padded]
        x = self.spatial_bn_act(x)     # [Batch, F1*D, 1, Time+padded]
        
        x = self.block2(x)             # [Batch, F2, 1, Time+padded]
        x = x.squeeze(2)               # [Batch, F2, Time+padded]
        x = self.output_proj(x)        # [Batch, 1, Time+padded]
        return x[..., :orig_len]

def print_summary():
    model = EEGNetS2()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    spatial_params = sum(p.numel() for p in model.spatial_attention.parameters() if p.requires_grad)
    print(f"EEGNetS2 Total Parameter Count: {params:,}")
    print(f"EEGNetS2 Spatial Parameter Count: {spatial_params:,}")
    
if __name__ == "__main__":
    print_summary()
