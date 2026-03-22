import torch
import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ

# ================= 1. ENCODER NETWORK =================
class KinematicEncoder(nn.Module):
    def __init__(self, input_dim=153, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            # Time: 8 -> 4
            nn.Conv1d(input_dim, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            # Time: 4 -> 2
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            # Time: 2 -> 1
            nn.Conv1d(512, latent_dim, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        # Permute axes: (Batch, Time, Features) -> (Batch, Features, Time)
        x = x.transpose(1, 2)
        z = self.encoder(x)
        # Flatten the last spatial dimension
        return z.squeeze(-1) # (Batch, latent_dim)

# ================= 2. DECODER NETWORK =================
class KinematicDecoder(nn.Module):
    def __init__(self, latent_dim=256, output_dim=153):
        super().__init__()
        self.decoder = nn.Sequential(
            # Time: 1 -> 2
            nn.ConvTranspose1d(latent_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            # Time: 2 -> 4
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            # Time: 4 -> 8
            nn.ConvTranspose1d(256, output_dim, kernel_size=4, stride=2, padding=1),
            # Final layer uses no activation to output physical values in range [-15.0, 15.0]
        )

    def forward(self, z):
        # Expand z to (Batch, latent_dim, 1) for ConvTranspose1d input
        z = z.unsqueeze(-1)
        x_reconstructed = self.decoder(z)
        # Permute back to original format: (Batch, Time, Features)
        return x_reconstructed.transpose(1, 2)

# ================= 3. COMPLETE GRFSQ-VAE SYSTEM =================
class GRFSQ_VAE(nn.Module):
    def __init__(self, input_dim=153, latent_dim=256, codebook_size=512, num_quantizers=3):
        super().__init__()
        
        self.encoder = KinematicEncoder(input_dim, latent_dim)
        
        # RVQ bottleneck (Residual Vector Quantizer)
        self.rvq = ResidualVQ(
            dim = latent_dim,
            num_quantizers = num_quantizers, # Number of output tokens (e.g., 3)
            codebook_size = codebook_size,   # Codebook size
            commitment_weight = 1.0,         # Weight to enforce latent commitment to codebook
            use_cosine_sim = True            # Cosine similarity often converges better
        )
        
        self.decoder = KinematicDecoder(latent_dim, input_dim)

    def forward(self, x):
        # 1. Encode physical matrix into latent space
        z = self.encoder(x)
        
        # 2. Quantization: map z to discrete IDs, then decode back to z_quantized
        # RVQ automatically returns commitment loss
        z_quantized, indices, commit_loss = self.rvq(z)
        
        # 3. Reconstruct motion
        x_recon = self.decoder(z_quantized)
        
        # Sum commitment loss across all RVQ layers
        commit_loss = commit_loss.sum()
        
        return x_recon, indices, commit_loss


# ================= TEST FULL SYSTEM ARCHITECTURE =================
if __name__ == "__main__":
    batch_size = 256
    dummy_input = torch.randn(batch_size, 8, 153)
    
    # Initialize model
    vae_model = GRFSQ_VAE(input_dim=153, latent_dim=256, codebook_size=512, num_quantizers=3)
    
    # Forward pass
    x_recon, token_indices, rvq_loss = vae_model(dummy_input)
    
    print("🚀 GRFSQ-VAE ARCHITECTURE TEST:")
    print(f"   -> Input shape       : {dummy_input.shape}")
    print(f"   -> Reconstruct shape : {x_recon.shape} (Must exactly match input)")
    print(f"   -> Discrete Tokens   : {token_indices.shape} (Expected: Batch x Num_Quantizers)")
    print(f"   -> RVQ Loss value    : {rvq_loss.item():.4f}")