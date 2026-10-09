#!/usr/bin/env bash
set -euo pipefail


REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${WORLDTS_PYTHON:-python}"
GPU="${GPU:-0}"
DATA_DIR="${WORLDTS_TEXT_DATA:-$REPO_ROOT/data}"
LLM_PATH="${WORLDTS_GPT2:-$REPO_ROOT/language_model/openai-community/gpt2}"
RUN_ID="${RUN_ID:-worldts_best_h12_frozen_$(date -u +%Y%m%dT%H%M%SZ)_pid$$}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$REPO_ROOT/checkpoints_${RUN_ID}}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/results_${RUN_ID}_test}"

if [[ -e "$CHECKPOINT_ROOT" || -e "$RESULT_ROOT" ]]; then
    echo "Refusing to reuse an existing run directory: $RUN_ID" >&2
    exit 2
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    mkdir -p "$CHECKPOINT_ROOT" "$RESULT_ROOT"
fi

run() {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf 'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '
        printf '%q ' "$@"
        printf '\n'
        return 0
    fi
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$@"
}

echo "Frozen WorldTS best H=12 rerun"
echo "  GPU: $GPU"
echo "  seed: 2025"
echo "  run id: $RUN_ID"
echo "  checkpoint root: $CHECKPOINT_ROOT"
echo "  result root: $RESULT_ROOT"

run "$PYTHON" -u -m exp.exp_worldts_forecasting \
  --is_training 1 --eval_split test --model WorldTS \
  --setting "${RUN_ID}_Agriculture_agri_i11_f29f3f8d0406" --model_id Agriculture_agri_i11_f29f3f8d0406 \
  --root_path "$DATA_DIR" --data_path Agriculture.csv --freq m \
  --seq_len 8 --label_len 4 --pred_len 12 --seed 2025 --gpu "$GPU" \
  --batch_size 32 --num_workers 0 --stage1_epochs 20 --stage2_epochs 20 \
  --stage1_lr 0.0009 --stage2_lr 0.0007 --patience 5 --loss MAE --use_revin 1 \
  --use_modal_condition 1 --use_time_condition 0 \
  --llm_path "$LLM_PATH" --llm_dim 768 --llm_layers 6 \
  --pool_type avg --text_embedding_batch_size 128 \
  --latent_dim 512 --modal_feature_dim 512 --modal_hidden_dim 256 \
  --state_encoder causal_patch --state_patch_len 2 --state_hidden_multiplier 1 \
  --predictor_patch_len 8 --predictor_patch_stride 4 --predictor_d_model 32 --predictor_n_heads 4 \
  --predictor_e_layers 1 --predictor_d_ff 64 --predictor_dropout 0 \
  --predictor_activation gelu --predictor_factor 1 --modal_gate_init_logit 0 \
  --cfa_reduction_factor 4 --cfa_dropout 0.3 --cfa_activation gelu \
  --latent_alignment_loss mae --latent_mse_weight 10 --latent_cosine_weight 15 \
  --latent_sigreg_weight 0.1 --latent_detach_target 1 --sigreg_std_weight 1 \
  --sigreg_cov_weight 0.04 --sigreg_eps 1e-4 \
  --decoder_type causal_patch --decoder_patch_len 2 --decoder_hidden_multiplier 1 \
  --decoder_warmup_weight 0.3 --checkpoints "$CHECKPOINT_ROOT" --results "$RESULT_ROOT"

run "$PYTHON" -u -m exp.exp_worldts_forecasting \
  --is_training 1 --eval_split test --model WorldTS \
  --setting "${RUN_ID}_Climate_h12_sym_e1_p2_h1_m3_o5_b32_s1lr0009_s2lr0003" \
  --model_id Climate_h12_sym_e1_p2_h1_m3_o5_b32_s1lr0009_s2lr0003 \
  --root_path "$DATA_DIR" --data_path Climate.csv --freq m \
  --seq_len 8 --label_len 4 --pred_len 12 --seed 2025 --gpu "$GPU" \
  --batch_size 32 --num_workers 0 --stage1_epochs 20 --stage2_epochs 20 \
  --stage1_lr 0.0009 --stage2_lr 0.0003 --patience 5 --loss MAE --use_revin 1 \
  --use_modal_condition 1 --use_time_condition 0 \
  --llm_path "$LLM_PATH" --llm_dim 768 --llm_layers 6 \
  --pool_type avg --text_embedding_batch_size 128 \
  --latent_dim 128 --modal_feature_dim 128 --modal_hidden_dim 512 \
  --state_encoder causal_patch --state_patch_len 2 --state_hidden_multiplier 1 \
  --predictor_patch_len 2 --predictor_patch_stride 1 --predictor_d_model 128 --predictor_n_heads 4 \
  --predictor_e_layers 3 --predictor_d_ff 256 --predictor_dropout 0 \
  --predictor_activation gelu --predictor_factor 1 --modal_gate_init_logit 0 \
  --cfa_reduction_factor 2 --cfa_dropout 0.1 --cfa_activation gelu \
  --latent_alignment_loss mae --latent_mse_weight 10 --latent_cosine_weight 15 \
  --latent_sigreg_weight 0.1 --latent_detach_target 1 --sigreg_std_weight 1 \
  --sigreg_cov_weight 0.04 --sigreg_eps 1e-4 \
  --decoder_type causal_patch --decoder_patch_len 2 --decoder_hidden_multiplier 1 \
  --decoder_warmup_weight 0.3 --checkpoints "$CHECKPOINT_ROOT" --results "$RESULT_ROOT"

run "$PYTHON" -u -m exp.exp_worldts_forecasting \
  --is_training 1 --eval_split test --model WorldTS \
  --setting "${RUN_ID}_Economy_sort_h12_detp1sym_localfine_b32_do025_s1lr0003_s2lr0002" \
  --model_id Economy_sort_h12_detp1sym_localfine_b32_do025_s1lr0003_s2lr0002 \
  --root_path "$DATA_DIR" --data_path Economy_sort.csv --freq m \
  --seq_len 8 --label_len 4 --pred_len 12 --seed 2025 --gpu "$GPU" \
  --batch_size 32 --num_workers 0 --stage1_epochs 20 --stage2_epochs 20 \
  --stage1_lr 0.0003 --stage2_lr 0.0002 --patience 5 --loss MAE --use_revin 1 \
  --use_modal_condition 1 --use_time_condition 0 \
  --llm_path "$LLM_PATH" --llm_dim 768 --llm_layers 6 \
  --pool_type avg --text_embedding_batch_size 128 \
  --latent_dim 256 --modal_feature_dim 256 --modal_hidden_dim 256 \
  --state_encoder causal_patch --state_patch_len 1 --state_hidden_multiplier 1 \
  --predictor_patch_len 2 --predictor_patch_stride 1 --predictor_d_model 64 --predictor_n_heads 4 \
  --predictor_e_layers 2 --predictor_d_ff 128 --predictor_dropout 0 \
  --predictor_activation gelu --predictor_factor 1 --modal_gate_init_logit 0 \
  --cfa_reduction_factor 4 --cfa_dropout 0.025 --cfa_activation gelu \
  --latent_alignment_loss mae --latent_mse_weight 10 --latent_cosine_weight 15 \
  --latent_sigreg_weight 0.1 --latent_detach_target 1 --sigreg_std_weight 1 \
  --sigreg_cov_weight 0.04 --sigreg_eps 1e-4 \
  --decoder_type causal_patch --decoder_patch_len 1 --decoder_hidden_multiplier 1 \
  --decoder_warmup_weight 0.3 --checkpoints "$CHECKPOINT_ROOT" --results "$RESULT_ROOT"

run "$PYTHON" -u -m exp.exp_worldts_forecasting \
  --is_training 1 --eval_split test --model WorldTS \
  --setting "${RUN_ID}_Energy_h12_mae_LR_lr_s2_1e4_aac3b9168d" \
  --model_id Energy_h12_mae_LR_lr_s2_1e4_aac3b9168d \
  --root_path "$DATA_DIR" --data_path Energy.csv --freq w \
  --seq_len 36 --label_len 18 --pred_len 12 --seed 2025 --gpu "$GPU" \
  --batch_size 32 --num_workers 0 --stage1_epochs 20 --stage2_epochs 20 \
  --stage1_lr 0.001 --stage2_lr 0.0001 --patience 5 --loss MAE --use_revin 1 \
  --use_modal_condition 1 --use_time_condition 0 \
  --llm_path "$LLM_PATH" --llm_dim 768 --llm_layers 6 \
  --pool_type avg --text_embedding_batch_size 128 \
  --latent_dim 256 --modal_feature_dim 256 --modal_hidden_dim 256 \
  --state_encoder causal_patch --state_patch_len 3 --state_hidden_multiplier 1 \
  --predictor_patch_len 4 --predictor_patch_stride 2 --predictor_d_model 64 --predictor_n_heads 4 \
  --predictor_e_layers 1 --predictor_d_ff 128 --predictor_dropout 0 \
  --predictor_activation gelu --predictor_factor 1 --modal_gate_init_logit 0 \
  --cfa_reduction_factor 4 --cfa_dropout 0.1 --cfa_activation gelu \
  --latent_alignment_loss mae --latent_mse_weight 10 --latent_cosine_weight 15 \
  --latent_sigreg_weight 0.1 --latent_detach_target 1 --sigreg_std_weight 1 \
  --sigreg_cov_weight 0.04 --sigreg_eps 1e-4 \
  --decoder_type causal_patch --decoder_patch_len 3 --decoder_hidden_multiplier 1 \
  --decoder_warmup_weight 0.3 --checkpoints "$CHECKPOINT_ROOT" --results "$RESULT_ROOT"

run "$PYTHON" -u -m exp.exp_worldts_forecasting \
  --is_training 1 --eval_split test --model WorldTS \
  --setting "${RUN_ID}_Health_h12_sym_roundlr_1e3_1e3" --model_id Health_h12_sym_roundlr_1e3_1e3 \
  --root_path "$DATA_DIR" --data_path Health.csv --freq w \
  --seq_len 36 --label_len 18 --pred_len 12 --seed 2025 --gpu "$GPU" \
  --batch_size 32 --num_workers 0 --stage1_epochs 20 --stage2_epochs 20 \
  --stage1_lr 0.001 --stage2_lr 0.001 --patience 5 --loss MAE --use_revin 1 \
  --use_modal_condition 1 --use_time_condition 0 \
  --llm_path "$LLM_PATH" --llm_dim 768 --llm_layers 6 \
  --pool_type avg --text_embedding_batch_size 128 \
  --latent_dim 256 --modal_feature_dim 256 --modal_hidden_dim 256 \
  --state_encoder causal_patch --state_patch_len 2 --state_hidden_multiplier 1 \
  --predictor_patch_len 2 --predictor_patch_stride 1 --predictor_d_model 64 --predictor_n_heads 4 \
  --predictor_e_layers 2 --predictor_d_ff 128 --predictor_dropout 0 \
  --predictor_activation gelu --predictor_factor 1 --modal_gate_init_logit 0 \
  --cfa_reduction_factor 4 --cfa_dropout 0 --cfa_activation gelu \
  --latent_alignment_loss mae --latent_mse_weight 10 --latent_cosine_weight 15 \
  --latent_sigreg_weight 0.1 --latent_detach_target 1 --sigreg_std_weight 1 \
  --sigreg_cov_weight 0.04 --sigreg_eps 1e-4 \
  --decoder_type causal_patch --decoder_patch_len 2 --decoder_hidden_multiplier 1 \
  --decoder_warmup_weight 0.3 --checkpoints "$CHECKPOINT_ROOT" --results "$RESULT_ROOT"

run "$PYTHON" -u -m exp.exp_worldts_forecasting \
  --is_training 1 --eval_split test --model WorldTS \
  --setting "${RUN_ID}_Security_sec_lc_f9480dc4ec12" --model_id Security_sec_lc_f9480dc4ec12 \
  --root_path "$DATA_DIR" --data_path Security.csv --freq m \
  --seq_len 8 --label_len 4 --pred_len 12 --seed 2025 --gpu "$GPU" \
  --batch_size 32 --num_workers 0 --stage1_epochs 20 --stage2_epochs 20 \
  --stage1_lr 0.0006 --stage2_lr 0.0001 --patience 5 --loss MAE --use_revin 1 \
  --use_modal_condition 1 --use_time_condition 0 \
  --llm_path "$LLM_PATH" --llm_dim 768 --llm_layers 6 \
  --pool_type avg --text_embedding_batch_size 128 \
  --latent_dim 128 --modal_feature_dim 128 --modal_hidden_dim 256 \
  --state_encoder causal_patch --state_patch_len 3 --state_hidden_multiplier 2 \
  --predictor_patch_len 8 --predictor_patch_stride 4 --predictor_d_model 32 --predictor_n_heads 4 \
  --predictor_e_layers 3 --predictor_d_ff 64 --predictor_dropout 0 \
  --predictor_activation gelu --predictor_factor 1 --modal_gate_init_logit 0 \
  --cfa_reduction_factor 4 --cfa_dropout 0.2 --cfa_activation gelu \
  --latent_alignment_loss mae --latent_mse_weight 10 --latent_cosine_weight 15 \
  --latent_sigreg_weight 0.1 --latent_detach_target 1 --sigreg_std_weight 1 \
  --sigreg_cov_weight 0.04 --sigreg_eps 1e-4 \
  --decoder_type causal_patch --decoder_patch_len 3 --decoder_hidden_multiplier 2 \
  --decoder_warmup_weight 0.3 --checkpoints "$CHECKPOINT_ROOT" --results "$RESULT_ROOT"

run "$PYTHON" -u -m exp.exp_worldts_forecasting \
  --is_training 1 --eval_split test --model WorldTS \
  --setting "${RUN_ID}_SocialGood_h12_detround_b32_s10005_s20005" \
  --model_id SocialGood_h12_detround_b32_s10005_s20005 \
  --root_path "$DATA_DIR" --data_path SocialGood.csv --freq m \
  --seq_len 8 --label_len 4 --pred_len 12 --seed 2025 --gpu "$GPU" \
  --batch_size 32 --num_workers 0 --stage1_epochs 20 --stage2_epochs 20 \
  --stage1_lr 0.0005 --stage2_lr 0.0005 --patience 5 --loss MAE --use_revin 1 \
  --use_modal_condition 1 --use_time_condition 0 \
  --llm_path "$LLM_PATH" --llm_dim 768 --llm_layers 6 \
  --pool_type avg --text_embedding_batch_size 128 \
  --latent_dim 256 --modal_feature_dim 256 --modal_hidden_dim 512 \
  --state_encoder causal_patch --state_patch_len 3 --state_hidden_multiplier 2 \
  --predictor_patch_len 8 --predictor_patch_stride 4 --predictor_d_model 64 --predictor_n_heads 4 \
  --predictor_e_layers 2 --predictor_d_ff 128 --predictor_dropout 0 \
  --predictor_activation gelu --predictor_factor 1 --modal_gate_init_logit 0 \
  --cfa_reduction_factor 4 --cfa_dropout 0.1 --cfa_activation gelu \
  --latent_alignment_loss mae --latent_mse_weight 10 --latent_cosine_weight 15 \
  --latent_sigreg_weight 0.1 --latent_detach_target 1 --sigreg_std_weight 1 \
  --sigreg_cov_weight 0.04 --sigreg_eps 1e-4 \
  --decoder_type causal_patch --decoder_patch_len 3 --decoder_hidden_multiplier 2 \
  --decoder_warmup_weight 0.3 --checkpoints "$CHECKPOINT_ROOT" --results "$RESULT_ROOT"

run "$PYTHON" -u -m exp.exp_worldts_forecasting \
  --is_training 1 --eval_split test --model WorldTS \
  --setting "${RUN_ID}_Traffic_traffic_i11_lr_fe210f669a3f" --model_id Traffic_traffic_i11_lr_fe210f669a3f \
  --root_path "$DATA_DIR" --data_path Traffic.csv --freq m \
  --seq_len 8 --label_len 4 --pred_len 12 --seed 2025 --gpu "$GPU" \
  --batch_size 8 --num_workers 0 --stage1_epochs 40 --stage2_epochs 40 \
  --stage1_lr 0.000095 --stage2_lr 0.0004 --patience 5 --loss MAE --use_revin 1 \
  --use_modal_condition 1 --use_time_condition 0 \
  --llm_path "$LLM_PATH" --llm_dim 768 --llm_layers 6 \
  --pool_type avg --text_embedding_batch_size 128 \
  --latent_dim 64 --modal_feature_dim 64 --modal_hidden_dim 256 \
  --state_encoder causal_patch --state_patch_len 7 --state_hidden_multiplier 4 \
  --predictor_patch_len 4 --predictor_patch_stride 2 --predictor_d_model 32 --predictor_n_heads 4 \
  --predictor_e_layers 2 --predictor_d_ff 64 --predictor_dropout 0 \
  --predictor_activation gelu --predictor_factor 1 --modal_gate_init_logit 0 \
  --cfa_reduction_factor 2 --cfa_dropout 0.1 --cfa_activation gelu \
  --latent_alignment_loss mae --latent_mse_weight 10 --latent_cosine_weight 15 \
  --latent_sigreg_weight 0.1 --latent_detach_target 1 --sigreg_std_weight 1 \
  --sigreg_cov_weight 0.04 --sigreg_eps 1e-4 \
  --decoder_type causal_patch --decoder_patch_len 7 --decoder_hidden_multiplier 4 \
  --decoder_warmup_weight 0.3 --checkpoints "$CHECKPOINT_ROOT" --results "$RESULT_ROOT"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "Previewed all 8 commands; no training was started."
else
    echo "All 8 runs completed. Results: $RESULT_ROOT"
fi
