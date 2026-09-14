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
#SBATCH --job-name=qwen-paper
#SBATCH --output=qwen-paper_%j.log

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

# Advanced task: the paper does not disclose these choices; specify them explicitly.
: "${LAMBDA_CUBE:?Set the cube loss coefficient}"
: "${LAMBDA_YAW:?Set the yaw loss coefficient}"
: "${PAPER_LR_SCHEDULE:?Choose constant, repo-step, or repo-epoch}"
: "${PAPER_MASK_REDUCTION:?Choose full or valid}"
OUTPUT="qwen_pano/outputs/pano_paper_${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)_$$}"
python -u -m qwen_pano.train \
  --profile paper --model "$MODEL" --workers 25 \
  --panorama-manifest qwen_pano/data/polished/train.jsonl \
  --perspective-manifest qwen_pano/data/projected_perspective/perspective.jsonl \
  --text-cache qwen_pano/cache/text_2512_mix \
  --lambda-cube "$LAMBDA_CUBE" --lambda-yaw "$LAMBDA_YAW" \
  --paper-lr-schedule "$PAPER_LR_SCHEDULE" --paper-mask-reduction "$PAPER_MASK_REDUCTION" \
  --output "$OUTPUT" "$@"
