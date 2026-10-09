#!/usr/bin/env bash
set -euo pipefail


REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
PYTHON_BIN="${WORLDTS_PYTHON:-python}"
GPU="${GPU:-0}"
RUN_NAME="${RUN_NAME:-worldts_best_mmsp_h24_seed2025_$(date -u +%Y%m%dT%H%M%SZ)_pid$$}"
RUN_ROOT="${RUN_ROOT:-$REPO_DIR/runs/$RUN_NAME}"
RESULT_DIR="$RUN_ROOT/result"

export DATA_PATH="${DATA_PATH:-$REPO_DIR/unica_datasets}"
export MODEL_PATH="${MODEL_PATH:-$REPO_DIR/pretrained_models}"
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

command=(
  "$PYTHON_BIN" "$REPO_DIR/main.py"
  --model_name ts_adapter/worldts_image
  --datasets mmsp
  --with_future
  --indexed_sample
  --split_val
  --num_workers 0
  --num_batches_per_epoch 20
  --sampler random
  --batch_size 8
  --gradient_clip 0.1
  --weight_decay 0
  --d_multi_modal 4
  --homogenizer_type linear
  --seq_len 2048
  --image_hidden_dim 768
  --modal_hidden_dim 256
  --modal_feature_dim 128
  --latent_dim 128
  --condition_dim 64
  --state_encoder causal_patch
  --state_patch_len 3
  --state_hidden_multiplier 2
  --decoder_type causal_patch
  --decoder_patch_len 3
  --decoder_hidden_multiplier 2
  --predictor_patch_len 64
  --predictor_patch_stride 32
  --predictor_d_model 64
  --predictor_n_heads 4
  --predictor_e_layers 2
  --predictor_d_ff 128
  --predictor_dropout 0
  --predictor_activation gelu
  --predictor_factor 1
  --modal_gate_init_logit -4.0
  --cfa_reduction_factor 8
  --cfa_dropout 0
  --cfa_activation gelu
  --use_revin 1
  --revin_eps 1e-5
  --latent_alignment_loss mae
  --latent_mse_weight 10
  --latent_cosine_weight 15
  --latent_sigreg_weight 0.1
  --latent_detach_target 1
  --sigreg_std_weight 1.0
  --sigreg_cov_weight 0.04
  --sigreg_eps 1e-4
  --stage1_epochs 20
  --stage2_epochs 20
  --stage1_lr 2.5e-4
  --stage2_lr 2e-4
  --patience 5
  --loss mae
  --decoder_warmup_weight 0.3
  --seed 2025
  --output_dir "$RESULT_DIR"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DATA_PATH=%q MODEL_PATH=%q ' "$DATA_PATH" "$MODEL_PATH"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

if [[ -e "$RUN_ROOT" ]]; then
  printf 'Refusing to reuse an existing run directory: %s\n' "$RUN_ROOT" >&2
  exit 2
fi

mkdir -p "$RESULT_DIR"
CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 "${command[@]}"

if [[ ! -s "$RESULT_DIR/normalized_results.csv" ]]; then
  printf 'Missing normalized test report: %s\n' "$RESULT_DIR/normalized_results.csv" >&2
  exit 1
fi
printf 'Normalized test report: %s\n' "$RESULT_DIR/normalized_results.csv"
