#!/bin/bash
for i in 0 2
do
    OFFSET=$((i * 40))
    JOB_NAME="mb_lift_chunk_$i"

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
#SBATCH --time=4:00:00        # 4-hour limit for fast scheduling priority
#SBATCH --output=logs/mb_chunk_${i}_%j.log

source setup_motionbert.sh
export HF_DATASETS_OFFLINE=1

# 160 GPUs scan in parallel — each lifts a video to 3D as soon as its 2D is ready
srun python -u pipeline/phase2_motionbert_gpu.py --offset $OFFSET --total_workers 200
EOT
done
