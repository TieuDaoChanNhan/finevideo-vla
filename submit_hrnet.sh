#!/bin/bash
# Chạy vòng lặp để submit 4 job, mỗi job 10 nodes (40 GPUs)
for i in {4..4}
do
    OFFSET=$((i * 40))
    JOB_NAME="hrnet_chunk_$i"
    
    echo "Submitting $JOB_NAME with offset $OFFSET..."
    
    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=18
#SBATCH --time=2:00:00
#SBATCH --output=logs/chunk_${i}_%j.log

source setup_hrnet_gpu.sh

mkdir -p logs outputs/2d_json workspace_temp

export TORCH_CUDA_ARCH_LIST="9.0"
export HF_DATASETS_OFFLINE=1

# Truyền offset vào để các job chia việc chuẩn 0-159
srun python -u phase1_hrnet_gpu.py --offset $OFFSET --total_workers 200
EOT
done