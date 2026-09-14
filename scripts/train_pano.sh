#!/bin/bash
#SBATCH -p debug
#SBATCH --nodelist=GPU4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=26
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --time=72:00:00
#SBATCH --mem=70G
#SBATCH --job-name=qwen-train
#SBATCH --output=qwen_pano/outputs/qwen-train_%j.log

set -eo pipefail
# Activate the named environment; no fixed Conda installation path.
eval "$("${CONDA_EXE:-conda}" shell.bash hook)"
conda activate qwen360
set -u

# Slurm executes a copied script. Locate the project from the submission directory.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f qwen_pano/train.py ]]; do
  [[ "$PWD" != / ]] || { echo "Submit from inside the DIT360 project." >&2; exit 2; }
  cd ..
done
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
MODEL="${QWEN_MODEL:-Qwen/Qwen-Image-2512}"
echo "Job ${SLURM_JOB_ID:-local}; node=$(hostname); Python=$(command -v python)"

# A fresh Slurm job gets a fresh directory; never overwrite an earlier run.
OUTPUT="qwen_pano/outputs/pano_official_full_${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)_$$}"
echo "Training output: $OUTPUT"
python -u -m qwen_pano.train \
  --profile repo-panorama --official-full-training-only \
  --model "$MODEL" \
  --panorama-manifest qwen_pano/data/cached_official_full/train.jsonl \
  --text-cache qwen_pano/cache/text_2512_official_full \
  --height 1024 --epochs 25 --batch-size 1 --accumulation-steps 4 \
  --learning-rate 5e-5 --rank 64 --lora-alpha 64 --workers 25 \
  --padding-columns 1 --lambda-cube 0.5 --lambda-yaw 0.5 \
  --output "$OUTPUT" "$@"
