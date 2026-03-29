#!/bin/bash

# --- BƯỚC 1: CÔ LẬP MÔI TRƯỜNG (CỰC KỲ QUAN TRỌNG) ---
# Xóa sạch đường dẫn thư viện hệ thống để tránh lỗi "Permission denied" 
# hoặc xung đột phiên bản Torch khi chạy trên node Booster.
source /e/project1/reformo/nguyen38/3d-human-pose/miniforge3/bin/activate
unset PYTHONPATH
unset PYTHONHOME

# --- BƯỚC 2: KÍCH HOẠT CONDA ---
# Đường dẫn này mặc định theo cấu hình Miniforge của cậu
conda activate vjepa_final

# --- BƯỚC 3: CẤU HÌNH GPU GH200 ---
# Ép hệ thống ưu tiên sử dụng nhân Flash Attention tích hợp trong Torch 2.5.1
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# --- BƯỚC 4: KIỂM TRA NHANH (SANITY CHECK) ---
# Chỉ in ra khi chạy trực tiếp, giúp cậu yên tâm trước khi submit job lớn
if [[ $- == *i* ]]; then
    python -c "import torch; print(f'🔥 Torch: {torch.__version__} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NOT FOUND\"}')"
fi