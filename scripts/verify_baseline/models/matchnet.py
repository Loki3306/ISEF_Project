import torch
import torch.nn as nn
import torch.nn.functional as F
from models.eegnet import EEGNet
from models.atcnet import ATCNet
from models.eegnet_tcn import EEGNetTCN
from models.eegnet_multiscale import EEGNetMultiScaleM2

class AudioEncoder(nn.Module):
    """
    Encodes 28-band Gammatone subbands into a latent representation.
    """
    def __init__(self, in_channels=28, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(64, latent_dim, kernel_size=1)
        )
        
    def forward(self, x):
        return self.net(x)

def _compute_mean_cosine_similarity(z_1, z_2):
    """Normalized cosine similarity averaged over the temporal dimension (dim=2)."""
    # F.cosine_similarity(dim=1) yields [B, T], which we average over time
    sim = F.cosine_similarity(z_1, z_2, dim=1)
    return sim.mean(dim=1)

class ContrastiveMatchNet(nn.Module):
    """
    A Siamese network that explicitly learns a matching function between EEG and Audio.
    """
    def __init__(self, eeg_model_type="eegnet", eeg_channels=8, audio_channels=28, latent_dim=64, num_subjects=17):
        super().__init__()
        
        # 1. EEG Encoder
        if eeg_model_type.lower() == "eegnet":
            # EEGNet outputs shape [B, 1, T] by default. We need [B, latent_dim, T]
            self.eeg_encoder = EEGNet(in_channels=eeg_channels)
            # Override the final projection
            self.eeg_encoder.output_proj = nn.Conv1d(16, latent_dim, kernel_size=1) # F2=16 by default
        elif eeg_model_type.lower() == "atcnet":
            self.eeg_encoder = ATCNet(in_channels=eeg_channels)
            # Override the final projection
            hidden_dim = 16 * 2 # F1 * D
            self.eeg_encoder.output_proj = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)
        elif eeg_model_type.lower() == "eegnet_tcn":
            self.eeg_encoder = EEGNetTCN(in_channels=eeg_channels)
            # Override the final projection. EEGNetTCN output_proj takes F2 channels (default 16)
            self.eeg_encoder.output_proj = nn.Conv1d(16, latent_dim, kernel_size=1)
        elif eeg_model_type.lower() == "eegnet_s1":
            from models.eegnet_s1 import EEGNetS1
            self.eeg_encoder = EEGNetS1(in_channels=eeg_channels)
            # Override the final projection
            self.eeg_encoder.output_proj = nn.Conv1d(16, latent_dim, kernel_size=1)
        elif eeg_model_type.lower() == "eegnet_s2":
            from models.eegnet_s2 import EEGNetS2
            self.eeg_encoder = EEGNetS2(in_channels=eeg_channels)
            # Override the final projection
            self.eeg_encoder.output_proj = nn.Conv1d(16, latent_dim, kernel_size=1)
        elif eeg_model_type.lower() == "eegnet_multiscale_m2":
            from models.eegnet_multiscale import EEGNetMultiScaleM2
            self.eeg_encoder = EEGNetMultiScaleM2(in_channels=eeg_channels)
            self.eeg_encoder.output_proj = nn.Conv1d(16, latent_dim, kernel_size=1)
        elif eeg_model_type.lower() == "sincalignnet":
            from models.sincalignnet import SincAlignNet
            self.eeg_encoder = SincAlignNet(in_channels=eeg_channels)
            self.eeg_encoder.output_proj = nn.Conv1d(16, latent_dim, kernel_size=1)
        elif eeg_model_type.lower() == "msca":
            from models.msca_eeg_encoder import MSCA_EEGEncoder
            self.eeg_encoder = MSCA_EEGEncoder(in_channels=eeg_channels)
            self.eeg_encoder.output_proj = nn.Conv1d(64, latent_dim, kernel_size=1)
        else:
            raise ValueError(f"Unknown eeg_model_type: {eeg_model_type}")
            
        # 2. Audio Encoder
        self.audio_encoder = AudioEncoder(in_channels=audio_channels, latent_dim=latent_dim)
        
        # 3. Domain Adversarial Components
        try:
            from models.msca_modules import GradientReversalLayer
            self.grl = GradientReversalLayer(lambda_=0.0)
        except ImportError:
            # Fallback if not running in the right directory context
            class _GRLFallback(nn.Module):
                def __init__(self, lambda_=0.0):
                    super().__init__()
                    self.lambda_ = lambda_
                def forward(self, x):
                    return x # DANN won't work in this fallback
            self.grl = _GRLFallback(lambda_=0.0)
            
        self.subject_classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim // 2, num_subjects)
        )
        
        self.latent_dim = latent_dim

    def encode_eeg(self, eeg):
        """ Returns [B, latent_dim, Time] """
        return self.eeg_encoder(eeg)
        
    def encode_audio(self, audio):
        """ Returns [B, latent_dim, Time] """
        return self.audio_encoder(audio)

    def forward(self, eeg, audio_a, audio_b, return_subject_logits=False):
        """
        Forward pass for training.
        eeg: [B, C, T]
        audio_a: [B, 28, T]
        audio_b: [B, 28, T]
        
        Returns latent representations (retaining simple 3-tensor signature for backward compatibility).
        """
        z_eeg = self.encode_eeg(eeg)
        z_a = self.encode_audio(audio_a)
        z_b = self.encode_audio(audio_b)
        
        if return_subject_logits:
            z_pool = z_eeg.mean(dim=-1)
            subj_feat = self.grl(z_pool)
            subj_logits = self.subject_classifier(subj_feat)
            return z_eeg, z_a, z_b, subj_logits
            
        return z_eeg, z_a, z_b

    def compute_similarities(self, z_eeg, z_a, z_b):
        """Shared similarity computation path."""
        sim_a = _compute_mean_cosine_similarity(z_eeg, z_a)
        sim_b = _compute_mean_cosine_similarity(z_eeg, z_b)
        return sim_a, sim_b

    def get_confidence_from_latents(self, z_eeg, z_a, z_b, temperature=1.0):
        """
        Computes a confidence score based on the decision margin.
        This score is a probability proxy scaled via temperature, but is uncalibrated.
        """
        sim_a, sim_b = self.compute_similarities(z_eeg, z_a, z_b)
        margin = sim_a - sim_b
        confidence = torch.sigmoid(margin / temperature)
        return confidence

    def get_confidence(self, eeg, audio_a, audio_b, temperature=1.0):
        """
        Computes a confidence score based on the difference in cosine similarity 
        between z_eeg and the two audio candidates.
        """
        z_eeg, z_a, z_b = self(eeg, audio_a, audio_b)
        return self.get_confidence_from_latents(z_eeg, z_a, z_b, temperature=temperature)

def contrastive_loss(z_eeg, z_a, z_b, margin=0.1):
    """
    Computes a max-margin contrastive loss based on cosine similarity.
    We assume z_a is the attended audio, and z_b is the unattended audio.
    """
    sim_a_mean = _compute_mean_cosine_similarity(z_eeg, z_a)
    sim_b_mean = _compute_mean_cosine_similarity(z_eeg, z_b)
    
    loss = F.relu(margin - (sim_a_mean - sim_b_mean)).mean()
    
    return loss, sim_a_mean.mean(), sim_b_mean.mean()

def anchored_contrastive_loss(z_eeg, z_a, z_b, margin=0.1, lambda_align=0.5, align_target=0.1):
    """
    Computes a max-margin contrastive loss while explicitly anchoring r_A to be positive.
    Penalizes if sim_a_mean falls below align_target.
    """
    sim_a_mean = _compute_mean_cosine_similarity(z_eeg, z_a)
    sim_b_mean = _compute_mean_cosine_similarity(z_eeg, z_b)
    
    l_margin = F.relu(margin - (sim_a_mean - sim_b_mean)).mean()
    l_align = F.relu(align_target - sim_a_mean).mean()
    
    loss = l_margin + lambda_align * l_align
    
    return loss, sim_a_mean.mean(), sim_b_mean.mean()

def infonce_loss(z_eeg, z_a, z_b, temperature=0.1):
    """
    Computes an InfoNCE loss across the batch.
    For each EEG representation, the network must identify the correct audio (z_a[i])
    out of 2B candidates: all z_a and all z_b in the batch.
    
    z_eeg, z_a, z_b shape: [B, latent_dim, T]
    """
    B, D, T = z_eeg.shape
    
    # 1. Normalize over the latent dimension
    z_eeg_norm = F.normalize(z_eeg, dim=1)
    z_a_norm = F.normalize(z_a, dim=1)
    z_b_norm = F.normalize(z_b, dim=1)
    
    # 2. Compute time-averaged similarity matrix
    # torch.einsum('bdt,cdt->bc', X, Y) computes the dot product over D and T for all pairs of B and C
    # We divide by T to get the mean similarity over time.
    sim_a = torch.einsum('bdt,cdt->bc', z_eeg_norm, z_a_norm) / T  # [B, B]
    sim_b = torch.einsum('bdt,cdt->bc', z_eeg_norm, z_b_norm) / T  # [B, B]
    
    # 3. Concatenate all candidates
    # The first B columns are comparisons against z_a (positive is on the diagonal)
    # The next B columns are comparisons against z_b (all are negative, including diagonal)
    logits = torch.cat([sim_a, sim_b], dim=1) / temperature  # [B, 2B]
    
    # 4. The correct target for eeg i is z_a i, which is at index i
    labels = torch.arange(B, device=logits.device)
    
    loss = F.cross_entropy(logits, labels)
    
    # For tracking purposes, return the mean similarity of the true positive and the hard negative
    with torch.no_grad():
        sim_a_diag = torch.diag(sim_a).mean()
        sim_b_diag = torch.diag(sim_b).mean()
        
    return loss, sim_a_diag, sim_b_diag

def dcca_loss(H1, H2, r1=1e-3, r2=1e-3, eps=1e-9):
    """
    Computes the Deep Canonical Correlation Analysis (DCCA) loss between two views.
    H1, H2: tensors of shape [Batch * T, Features] or [Batch, Features]
    """
    # 1. Mean center the batches
    H1_mean = H1 - H1.mean(dim=0, keepdim=True)
    H2_mean = H2 - H2.mean(dim=0, keepdim=True)
    
    batch_size = H1.size(0)
    
    # 2. Compute covariance matrices
    SigmaHat12 = (1.0 / (batch_size - 1)) * torch.matmul(H1_mean.t(), H2_mean)
    SigmaHat11 = (1.0 / (batch_size - 1)) * torch.matmul(H1_mean.t(), H1_mean) + r1 * torch.eye(H1.size(1), device=H1.device)
    SigmaHat22 = (1.0 / (batch_size - 1)) * torch.matmul(H2_mean.t(), H2_mean) + r2 * torch.eye(H2.size(1), device=H2.device)
    
    # 3. Compute inverse square roots using robust eigenvalue decomposition
    D1, V1 = torch.linalg.eigh(SigmaHat11)
    D1 = torch.clamp(D1, min=eps)
    SigmaHat11RootInv = torch.matmul(torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
    
    D2, V2 = torch.linalg.eigh(SigmaHat22)
    D2 = torch.clamp(D2, min=eps)
    SigmaHat22RootInv = torch.matmul(torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())
    
    # 4. Compute T = Σ11^{-1/2} Σ12 Σ22^{-1/2}
    T_matrix = torch.matmul(torch.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)
    
    # 5. Compute CCA objective: sum of singular values of T
    # Use svdvals for gradient stability
    svdvals = torch.linalg.svdvals(T_matrix)
    
    # Loss is the negative sum of canonical correlations
    loss = -torch.sum(svdvals)
    
    return loss, svdvals.mean(), svdvals.mean()

if __name__ == "__main__":
    model = ContrastiveMatchNet("eegnet")
    eeg = torch.randn(16, 8, 320)
    audio_a = torch.randn(16, 28, 320)
    audio_b = torch.randn(16, 28, 320)
    
    z_eeg, z_a, z_b = model(eeg, audio_a, audio_b)
    print("Z_eeg:", z_eeg.shape)
    loss, sa, sb = infonce_loss(z_eeg, z_a, z_b)
    print(f"InfoNCE Loss: {loss.item():.4f} | Sim A: {sa:.4f} | Sim B: {sb:.4f}")
