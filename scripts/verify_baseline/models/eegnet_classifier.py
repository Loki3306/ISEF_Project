import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet_Classifier(nn.Module):
    """
    Standard EEGNet adapted for binary classification.
    Takes [Batch, Channels, Time], outputs [Batch] logits for BCEWithLogitsLoss.
    Uses ELU instead of GELU to remain closer to original EEGNet paper, and adds intermediate pooling.
    """
    def __init__(self, in_channels=64, F1=8, D=2, F2=16, kernel_length=64, dropout_rate=0.25):
        super().__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length//2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (in_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )
        
        self.classifier = nn.Linear(F2, 1)

    def forward(self, x):
        # Input: [Batch, Channels, Time]
        orig_len = x.shape[-1]
        x = x.unsqueeze(1) # [Batch, 1, Channels, Time]
        
        # Because padding might add an extra element if kernel is even, we ensure size matches
        x = self.block1(x)
        x = self.block2(x)
        
        # Global Average Pooling over time dimension
        # Shape: [Batch, F2, 1, T_reduced]
        x = x.mean(dim=3).squeeze(2) # [Batch, F2]
        
        logits = self.classifier(x) # [Batch, 1]
        return logits.squeeze(1) # [Batch]

if __name__ == "__main__":
    model = EEGNet_Classifier()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EEGNet Classifier Parameter Count: {params:,}")
    dummy = torch.randn(4, 64, 320) # 5 seconds at 64Hz
    out = model(dummy)
    print(f"Output Shape: {out.shape}")
