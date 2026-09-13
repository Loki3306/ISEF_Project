import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sincalignnet_paper import SincAlignEEGEncoder
from models.matchnet import AudioEncoder

class SA2AudioEncoder(nn.Module):
    """
    Adapts the Gammatone AudioEncoder to output a single 128-D vector
    for the SA-2 diagnostic.
    """
    def __init__(self, in_channels=28, latent_dim=64):
        super().__init__()
        self.base_encoder = AudioEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(latent_dim, 128)
        
    def forward(self, x):
        # x: [B, 28, T]
        x = self.base_encoder(x) # [B, 64, T]
        x = self.pool(x) # [B, 64, 1]
        x = x.squeeze(-1) # [B, 64]
        x = self.proj(x) # [B, 128]
        return x

class SA2MatchNet(nn.Module):
    """
    SA-2 Diagnostic MatchNet for Vector-based Contrastive Alignment.
    """
    def __init__(self, eeg_channels=8, audio_channels=28, sample_rate=64, seq_len=640):
        super().__init__()
        
        self.eeg_encoder = SincAlignEEGEncoder(
            in_channels=eeg_channels, 
            sample_rate=sample_rate, 
            seq_len=seq_len
        )
        
        self.audio_encoder = SA2AudioEncoder(
            in_channels=audio_channels, 
            latent_dim=64
        )
        
    def encode_eeg(self, eeg):
        return self.eeg_encoder(eeg)
        
    def encode_audio(self, audio):
        return self.audio_encoder(audio)

    def forward(self, eeg, audio_a, audio_b):
        z_eeg = self.encode_eeg(eeg)
        z_a = self.encode_audio(audio_a)
        z_b = self.encode_audio(audio_b)
        return z_eeg, z_a, z_b

def infonce_vector_loss(z_eeg, z_a, z_b, temperature=0.1):
    """
    Computes an InfoNCE loss across the batch for 1D vectors.
    z_eeg, z_a, z_b shape: [B, 128]
    """
    B = z_eeg.shape[0]
    
    # Normalize vectors
    z_eeg_norm = F.normalize(z_eeg, dim=1)
    z_a_norm = F.normalize(z_a, dim=1)
    z_b_norm = F.normalize(z_b, dim=1)
    
    # Compute similarity matrices [B, B]
    sim_a = torch.matmul(z_eeg_norm, z_a_norm.T)
    sim_b = torch.matmul(z_eeg_norm, z_b_norm.T)
    
    logits = torch.cat([sim_a, sim_b], dim=1) / temperature  # [B, 2B]
    labels = torch.arange(B, device=logits.device)
    
    loss = F.cross_entropy(logits, labels)
    
    with torch.no_grad():
        sim_a_diag = torch.diag(sim_a).mean()
        sim_b_diag = torch.diag(sim_b).mean()
        
    return loss, sim_a_diag, sim_b_diag
