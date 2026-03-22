import torch
import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ

# ================= 1. MẠNG ENCODER =================
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
        # Đảo trục: (Batch, Time, Features) -> (Batch, Features, Time)
        x = x.transpose(1, 2)
        z = self.encoder(x)
        # Ép phẳng chiều không gian cuối cùng
        return z.squeeze(-1) # (Batch, latent_dim)

# ================= 2. MẠNG DECODER =================
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
            # Lớp cuối không dùng Activation để xuất ra các giá trị vật lý [-15.0, 15.0]
        )

    def forward(self, z):
        # Mở rộng z thành (Batch, latent_dim, 1) để đưa vào ConvTranspose1d
        z = z.unsqueeze(-1)
        x_reconstructed = self.decoder(z)
        # Đảo trục về lại nguyên bản: (Batch, Time, Features)
        return x_reconstructed.transpose(1, 2)

# ================= 3. HỆ THỐNG GRFSQ-VAE HOÀN CHỈNH =================
class GRFSQ_VAE(nn.Module):
    def __init__(self, input_dim=153, latent_dim=256, codebook_size=512, num_quantizers=3):
        super().__init__()
        
        self.encoder = KinematicEncoder(input_dim, latent_dim)
        
        # Nút thắt cổ chai RVQ (Residual Vector Quantizer)
        self.rvq = ResidualVQ(
            dim = latent_dim,
            num_quantizers = num_quantizers, # Số Token xuất ra (VD: 3)
            codebook_size = codebook_size,   # Kích thước Từ điển
            commitment_weight = 1.0,         # Trọng số ép Latent bám sát Codebook
            use_cosine_sim = True            # Dùng Cosine Similarity thường hội tụ tốt hơn
        )
        
        self.decoder = KinematicDecoder(latent_dim, input_dim)

    def forward(self, x):
        # 1. Mã hóa ma trận vật lý thành Không gian ẩn
        z = self.encoder(x)
        
        # 2. Lượng tử hóa: Ép z thành các ID rời rạc, rồi giải mã lại thành z_quantized
        # RVQ tự động nhả ra cho chúng ta Commitment Loss
        z_quantized, indices, commit_loss = self.rvq(z)
        
        # 3. Phục hồi lại chuyển động
        x_recon = self.decoder(z_quantized)
        
        # Hàm loss của Codebook phải được sum lại cho toàn bộ các layer RVQ
        commit_loss = commit_loss.sum()
        
        return x_recon, indices, commit_loss


# ================= TEST KIẾN TRÚC TOÀN HỆ THỐNG =================
if __name__ == "__main__":
    batch_size = 256
    dummy_input = torch.randn(batch_size, 8, 153)
    
    # Khởi tạo siêu mô hình
    vae_model = GRFSQ_VAE(input_dim=153, latent_dim=256, codebook_size=512, num_quantizers=3)
    
    # Chạy Forward Pass
    x_recon, token_indices, rvq_loss = vae_model(dummy_input)
    
    print("🚀 TEST KIẾN TRÚC GRFSQ-VAE:")
    print(f"   -> Input shape       : {dummy_input.shape}")
    print(f"   -> Reconstruct shape : {x_recon.shape} (Phải trùng khớp hoàn toàn với Input)")
    print(f"   -> Discrete Tokens   : {token_indices.shape} (Dự kiến: Batch x Num_Quantizers)")
    print(f"   -> RVQ Loss value    : {rvq_loss.item():.4f}")