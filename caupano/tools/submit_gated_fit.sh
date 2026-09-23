#!/bin/bash
#SBATCH -p debug
#SBATCH --nodelist=GPU3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=26
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --time=72:00:00
#SBATCH --mem=70G
#SBATCH --job-name=caupano-gated
#SBATCH --output=qwen_pano/outputs/caupano-gated-%j.log

set -eo pipefail

# Activate qwen360 environment.
eval "$("${CONDA_EXE:-conda}" shell.bash hook)"
conda activate qwen360
set -u

# Slurm executes a copied script. Locate the DIT360 project root
# from the submission directory.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f qwen_pano/train.py ]]; do
  [[ "$PWD" != / ]] || {
    echo "Submit from inside the DIT360 project." >&2
    exit 2
  }
  cd ..
done

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "Job ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Conda env: ${CONDA_DEFAULT_ENV:-unknown}"
echo "Python: $(command -v python)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

experiment=qwen_pano/outputs/caupano/experiments/gated_epoch005

echo "[1/3] branch regression checks (CPU)"
python -u -m unittest qwen_pano.caupano.tests.test_gated_world

if [[ "${1:-train}" == audit ]]; then
  echo "Read-only attention audit of run003; no training"
  python -u -m qwen_pano.caupano.sample_world \
    --experiment "$experiment" \
    --run "$experiment/fit_run003_budget" \
    --weights "$experiment/fit_run003_budget/world-step01000.safetensors" \
    --text-cache "$experiment/text_cache" \
    --output "$experiment/attention_run003" \
    --splits fit --per-split 2 --modes world --attention-report \
    --steps 28 --height 1024 --true-cfg-scale 4 --seed 42
  exit 0
fi

train_steps=1000
eval_every=250
case "${1:-baseline}" in
  expand32|multiscan)
    if [[ "$1" == multiscan ]]; then
      experiment=qwen_pano/outputs/caupano/experiments/gated_epoch005_multiscan
    else
      experiment=qwen_pano/outputs/caupano/experiments/gated_epoch005_train32
    fi
    run_name=fit_run001
    adapter_options=()
    train_steps=4000
    eval_every=1000
    python -u -m qwen_pano.cache_text \
      --model Qwen/Qwen-Image-2512 \
      --revision 25468b98e3276ca6700de15c6628e51b7de54a26 \
      --local-files-only \
      --manifests "$experiment/fit.jsonl" "$experiment/dev.jsonl" \
      --output "$experiment/text_cache" --device cuda:0 --max-sequence-length 512
    ;;
  baseline)
    run_name=fit_run004_baseline
    adapter_options=()
    ;;
  balanced)
    run_name=fit_run004_balanced
    adapter_options=(--balance-world-types)
    ;;
  *)
    echo "Usage: sbatch $0 [multiscan|expand32|baseline|balanced|audit]" >&2
    exit 2
    ;;
esac
echo "[2/3] $run_name; spatial prior disabled"

python -u -m qwen_pano.caupano.train_world \
  --experiment "$experiment" \
  --text-cache "$experiment/text_cache" \
  --output "$experiment/$run_name" \
  --adapter-kind gated "${adapter_options[@]}" \
  --residual-scale hidden_rms --adam-eps 1e-8 \
  --height 1024 \
  --steps "$train_steps" \
  --eval-every "$eval_every" \
  --eval-fractions 0.25 0.5 0.75 \
  --learning-rate 0.0001 \
  --seed 42

echo "[3/3] generate comparisons"
printf -v weight_file 'world-step%05d.safetensors' "$train_steps"

python -u -m qwen_pano.caupano.sample_world \
  --experiment "$experiment" \
  --run "$experiment/$run_name" \
  --weights "$experiment/$run_name/$weight_file" \
  --text-cache "$experiment/text_cache" \
  --output "$experiment/images_${run_name}_step${train_steps}" \
  --attention-report \
  --modes no_world world position_scrambled shuffled_world \
  --per-split 2 \
  --steps 28 \
  --height 1024 \
  --true-cfg-scale 4 \
  --seed 42

echo "Finished."
echo "Training output: $experiment/$run_name"
echo "Images: $experiment/images_${run_name}_step${train_steps}"
