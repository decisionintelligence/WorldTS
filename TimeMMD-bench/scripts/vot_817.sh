#!/usr/bin/env bash

# Clean rerun of the configuration that is actually enabled in scripts/vot.sh.
# The forecasting/model hyperparameters are unchanged. Only run isolation,
# output locations, and optional device selection are added.

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/opt/conda/envs/vot/bin/python}

MAX_PARALLEL=4
DEVICES="0,1,2,3,4,5"
RUN_ID=""
DRY_RUN=0

usage() {
    echo "Usage: bash scripts/vot_817.sh [options]"
    echo "  --max-parallel N    Parallel processes (default: 4)"
    echo "  --devices IDS       GPU candidates passed to run_clip.py"
    echo "                      (default: 0,1,2,3,4,5)"
    echo "  --run-id ID         Optional persistent run identifier"
    echo "  --dry-run           Print commands without launching"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-parallel)
            MAX_PARALLEL=$2
            shift 2
            ;;
        --devices)
            DEVICES=$2
            shift 2
            ;;
        --run-id)
            RUN_ID=$2
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
    echo "--max-parallel must be a positive integer" >&2
    exit 2
fi
if [[ ! "$DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "--devices must look like 6 or 6,7" >&2
    exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 2
fi
if [[ ! -f "$REPO_ROOT/data/SocialGood.csv" ]]; then
    echo "Dataset not found: $REPO_ROOT/data/SocialGood.csv" >&2
    exit 2
fi
if [[ ! -d "$REPO_ROOT/data/stat_predictions_rag/SocialGood" ]]; then
    echo "Released text predictions not found under data/stat_predictions_rag" >&2
    exit 2
fi

if [[ -z "$RUN_ID" ]]; then
    RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_pid$$"
fi
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "Invalid --run-id: $RUN_ID" >&2
    exit 2
fi

RUN_ROOT="$REPO_ROOT/A_result/vot_817/$RUN_ID"
CHECKPOINT_ROOT="$REPO_ROOT/checkpoints_vot_817/$RUN_ID"
LOG_ROOT="$RUN_ROOT/logs"
RESULT_ROOT="$RUN_ROOT/results"
DATA_VIEW="$RUN_ROOT/data_view"

# These are the values that actually execute in the released vot.sh.
PRED_LENS=(6 8 10 12)
SEED=2025
SEQ_LEN=8
LABEL_LEN=4
BATCH_SIZE=32
LEARNING_RATE=0.00001
LEARNING_RATE_CLIP=0.0001
LEARNING_RATE_CORR=$LEARNING_RATE
TRAIN_EPOCHS=10
TRAIN_EPOCHS_CLIP=10
TRAIN_EPOCHS_CORR=10
CORRECT=1
IF_LRADJ=1
CLIP_T=0.0
DROPOUT=0.1
DECOMP_W=0.5

echo "Clean VoT released-script rerun"
echo "  dataset: SocialGood"
echo "  horizons: ${PRED_LENS[*]}"
echo "  seed: $SEED"
echo "  Stage-3 LR: $LEARNING_RATE_CORR"
echo "  jobs: 4"
echo "  run id: $RUN_ID"
echo "  checkpoint root: $CHECKPOINT_ROOT"

if [[ "$DRY_RUN" -eq 0 ]]; then
    # Refuse to start from any pre-existing run/checkpoint path. This is the
    # guarantee that Stage 1, Stage 2, and Stage 3 cannot be reused.
    if [[ -e "$RUN_ROOT" ]]; then
        echo "Run directory already exists: $RUN_ROOT" >&2
        exit 3
    fi
    if [[ -e "$CHECKPOINT_ROOT" ]]; then
        echo "Checkpoint directory already exists: $CHECKPOINT_ROOT" >&2
        exit 3
    fi
    mkdir -p "$LOG_ROOT" "$RESULT_ROOT" "$CHECKPOINT_ROOT" "$DATA_VIEW"
    # The released loader asks for stat_predictions_ecnu_rag, while the
    # released archive contains stat_predictions_rag. Provide a run-local,
    # read-only-compatible view without changing Python/model source files.
    ln -s "$REPO_ROOT/data/SocialGood.csv" "$DATA_VIEW/SocialGood.csv"
    ln -s "$REPO_ROOT/data/stat_predictions_rag" \
        "$DATA_VIEW/stat_predictions_ecnu_rag"
    {
        echo "run_id=$RUN_ID"
        echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "dataset=SocialGood"
        echo "horizons=${PRED_LENS[*]}"
        echo "seed=$SEED"
        echo "checkpoint_reuse=false"
        echo "checkpoint_root=$CHECKPOINT_ROOT"
        echo "data_view=$DATA_VIEW"
        echo "released_prediction_alias=stat_predictions_ecnu_rag->stat_predictions_rag"
        echo "devices=$DEVICES"
        echo "git_commit=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    } > "$RUN_ROOT/manifest.txt"
fi

declare -a ACTIVE_PIDS=()
declare -A PID_TO_CONFIG=()
FAILURES=0
LAUNCHED=0

wait_for_batch() {
    local pid
    local config_id
    for pid in "${ACTIVE_PIDS[@]}"; do
        config_id=${PID_TO_CONFIG[$pid]}
        if wait "$pid"; then
            echo "[completed] $config_id"
        else
            echo "[failed] $config_id" >&2
            FAILURES=$((FAILURES + 1))
        fi
        unset "PID_TO_CONFIG[$pid]"
    done
    ACTIVE_PIDS=()
}

for pred_len in "${PRED_LENS[@]}"; do
    config_id="SocialGood_H${pred_len}_seed${SEED}_released"
    model_id="PatchTST_clip_SocialGood_seed_${SEED}_pred_${pred_len}_seq_${SEQ_LEN}_label_${LABEL_LEN}_bs_${BATCH_SIZE}_corr_${CORRECT}_lr_${LEARNING_RATE}_lr_clip_${LEARNING_RATE_CLIP}_lr_corr_${LEARNING_RATE_CORR}_e_${TRAIN_EPOCHS}_e_clip_${TRAIN_EPOCHS_CLIP}_e_corr_${TRAIN_EPOCHS_CORR}_if_lradj_${IF_LRADJ}_clip_t_${CLIP_T}_dt_${DROPOUT}_dw_${DECOMP_W}"
    log_file="$LOG_ROOT/${config_id}.log"
    result_file="$RESULT_ROOT/${config_id}.txt"

    cmd=(
        "$PYTHON_BIN" -u run_clip.py
        --task_name long_term_forecast
        --is_training 1
        --root_path "$DATA_VIEW"
        --data_path SocialGood.csv
        --model_id "$model_id"
        --model PatchTST_clip
        --data custom
        --seq_len "$SEQ_LEN"
        --label_len "$LABEL_LEN"
        --pred_len "$pred_len"
        --des Exp
        --seed "$SEED"
        --save_name "$result_file"
        --multimodal 1
        --correct "$CORRECT"
        --tr_sea 1
        --llm_model GPT2
        --batch_size "$BATCH_SIZE"
        --train_epochs "$TRAIN_EPOCHS"
        --train_epochs_clip "$TRAIN_EPOCHS_CLIP"
        --train_epochs_corr "$TRAIN_EPOCHS_CORR"
        --learning_rate "$LEARNING_RATE"
        --learning_rate_clip "$LEARNING_RATE_CLIP"
        --learning_rate_corr "$LEARNING_RATE_CORR"
        --if_lradj "$IF_LRADJ"
        --clip_t "$CLIP_T"
        --dropout "$DROPOUT"
        --decomp_w "$DECOMP_W"
        --patience 5
        --devices "$DEVICES"
        --checkpoints "$CHECKPOINT_ROOT"
    )

    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '[dry-run] CUDA_VISIBLE_DEVICES=%q ' "$DEVICES"
        printf '%q ' "${cmd[@]}"
        printf '\n'
        continue
    fi

    echo "[launching] $config_id"
    (
        cd "$REPO_ROOT" || exit 1
        CUDA_VISIBLE_DEVICES="$DEVICES" "${cmd[@]}" > "$log_file" 2>&1
    ) &
    pid=$!
    ACTIVE_PIDS+=("$pid")
    PID_TO_CONFIG[$pid]=$config_id
    LAUNCHED=$((LAUNCHED + 1))

    if [[ "${#ACTIVE_PIDS[@]}" -ge "$MAX_PARALLEL" ]]; then
        wait_for_batch
    fi
done

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run complete: printed 4 commands; nothing was launched."
    exit 0
fi

if [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; then
    wait_for_batch
fi

{
    echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "launched=$LAUNCHED"
    echo "failures=$FAILURES"
} >> "$RUN_ROOT/manifest.txt"

echo "Finished: launched=$LAUNCHED failures=$FAILURES"
echo "Results: $RESULT_ROOT"
echo "Logs: $LOG_ROOT"
exit "$FAILURES"
