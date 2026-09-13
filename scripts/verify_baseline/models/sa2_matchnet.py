import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sincalignnet_paper import SincAlignEEGEncoder
from models.matchnet import AudioEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sincalignnet_paper import SincAlignEEGEncoder
from models.matchnet import AudioEncoder

class SA2MatchNet(nn.Module):
    """
    SA-2 Diagnostic MatchNet for Sequence-based Contrastive Alignment.
    """
    def __init__(self, eeg_channels=8, audio_channels=28, sample_rate=64, seq_len=640, latent_dim=64):
        super().__init__()
        
        self.eeg_encoder = SincAlignEEGEncoder(
            in_channels=eeg_channels, 
            sample_rate=sample_rate, 
            seq_len=seq_len
        )
        
        self.audio_encoder = AudioEncoder(
            in_channels=audio_channels, 
            latent_dim=latent_dim
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
