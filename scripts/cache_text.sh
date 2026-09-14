#!/bin/bash
#SBATCH -p debug
#SBATCH --nodelist=GPU4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --time=72:00:00
#SBATCH --mem=70G
#SBATCH --job-name=qwen-text
#SBATCH --output=qwen-text_%j.log

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

# Only cache frozen text features; completed entries are reused.
python -u -m qwen_pano.cache_text \
  --model "$MODEL" \
  --manifests qwen_pano/data/cached_official_full/train.jsonl \
  --output qwen_pano/cache/text_2512_official_full "$@"
