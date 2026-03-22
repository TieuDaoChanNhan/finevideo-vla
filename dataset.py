import os
import glob
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class KinematicDataset(Dataset):
    def __init__(self, data_dir="outputs/states/", max_clip_val=15.0):
        """
        Initialize Dataset.
        Load all JSONL files, convert to Float32 (PyTorch standard), and store in RAM.
        """
        self.windows = []
        self.max_clip_val = max_clip_val
        
        jsonl_files = glob.glob(os.path.join(data_dir, '*_states.jsonl'))
        print(f"📦 Loading {len(jsonl_files)} files into memory...")
        
        for file in jsonl_files:
            try:
                with open(file, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        # Convert array to Float32 for optimal GPU processing
                        state = np.array(data["states"], dtype=np.float32)
                        
                        # Only keep chunks with correct shape (8, 153)
                        if state.shape == (8, 153):
                            self.windows.append(state)
            except Exception as e:
                print(f"⚠️ Skipping corrupted file {file}: {e}")
                
        self.total_samples = len(self.windows)
        print(f"✅ Successfully loaded {self.total_samples:,} kinematic chunks!")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # 1. Fetch NumPy array and convert to PyTorch Tensor
        x = torch.tensor(self.windows[idx])
        
        return x

# ================= RUN TEST (DATALOADER SANITY CHECK) =================
if __name__ == "__main__":
    # Initialize Dataset
    dataset = KinematicDataset(data_dir="outputs/states/")
    
    # Initialize DataLoader: load 256 chunks per batch, shuffle randomly,
    # use 4 worker processes (num_workers) for parallel data loading.
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
    
    # Fetch the first batch to inspect its structure
    for batch_idx, batch in enumerate(dataloader):
        print("\n🚀 FIRST BATCH LOADED TO GPU:")
        print(f"   -> Data type : {batch.dtype}")
        print(f"   -> Shape     : {batch.shape}")
        
        # Check extrema to ensure clamping works correctly
        print(f"   -> Max value : {batch.max().item():.2f}")
        print(f"   -> Min value : {batch.min().item():.2f}")
        break # Only print one batch and stop