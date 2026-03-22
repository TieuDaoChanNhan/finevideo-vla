import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import from the files you created
from dataset import KinematicDataset
from model import GRFSQ_VAE

# ================= 1. PHYSICS-BASED LOSS FUNCTION (WEIGHTED MSE) =================
def compute_kinematic_loss(x, x_recon, commit_loss):
    """
    x, x_recon shape: (Batch, 8, 153)
    pos: 0->51, vel: 51->102, acc: 102->153
    """
    # Compute MSE for each kinematic component
    pos_loss = torch.mean((x[..., 0:51] - x_recon[..., 0:51])**2)
    vel_loss = torch.mean((x[..., 51:102] - x_recon[..., 51:102])**2)
    acc_loss = torch.mean((x[..., 102:153] - x_recon[..., 102:153])**2)
    
    # Priority weights: Position > Velocity > Acceleration
    total_loss = (1.0 * pos_loss) + (0.5 * vel_loss) + (0.1 * acc_loss) + commit_loss
    
    return total_loss, pos_loss, vel_loss, acc_loss

# ================= 2. TRAINING SETUP =================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting training on device: {device}")

    # Initialize dataset
    dataset = KinematicDataset(data_dir="outputs/states/")
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, drop_last=True)
    
    # Initialize model (Latent = 128 as suggested for better stability, or keep 256 if preferred)
    model = GRFSQ_VAE(input_dim=153, latent_dim=128, codebook_size=512, num_quantizers=3).to(device)
    
    # Initialize optimizer (AdamW works very well for VAE)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    epochs = 50
    
    # ================= 3. TRAINING LOOP =================
    for epoch in range(1, epochs + 1):
        model.train()
        total_epoch_loss = 0.0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        
        for batch_idx, x in enumerate(pbar):
            x = x.to(device)
            
            # Clear old gradients
            optimizer.zero_grad()
            
            # Forward pass: feed through GRFSQ-VAE
            x_recon, tokens, commit_loss = model(x)
            
            # Compute loss
            loss, p_loss, v_loss, a_loss = compute_kinematic_loss(x, x_recon, commit_loss)
            
            # Backward pass: compute gradients
            loss.backward()
            
            # Prevent exploding gradients (Gradient Clipping - critical for VAE stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Update logs on progress bar
            total_epoch_loss += loss.item()
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Pos': f"{p_loss.item():.4f}",
                'Vel': f"{v_loss.item():.4f}"
            })
            
        avg_loss = total_epoch_loss / len(dataloader)
        print(f"🏁 Epoch {epoch} finished | Avg Loss: {avg_loss:.4f}\n")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"outputs/grfsq_vae_epoch_{epoch}.pth")
            print(f"💾 Model checkpoint saved at epoch {epoch}")

if __name__ == "__main__":
    train()