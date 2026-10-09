#!/bin/bash

# Parse command line arguments
MAX_PARALLEL=8  # Default value
while [[ $# -gt 0 ]]; do
  case $1 in
    --max-parallel)
      MAX_PARALLEL="$2"
      shift 2
      ;;
    *)
      # Skip unknown arguments
      shift
      ;;
  esac
done

echo "Setting maximum parallel jobs to: $MAX_PARALLEL"

# Create logs directory
log_path="A_result/results_stat"
mkdir -p $log_path

# Define datasets and their corresponding pred_len values
declare -A datasets=(
#    ["Agriculture"]="6 8 10 12"
#    ["Climate"]="6 8 10 12"
#    ["Economy"]="6 8 10 12"
#    ["Energy"]="12 24 36 48"
#    ["Environment"]="48 96 192 336"
#    ["Health"]="12 24 36 48"
#    ["Security"]="6 8 10 12"
    ["SocialGood"]="6 8 10 12"
#    ["Traffic"]="6 8 10 12"
#    ["new_metr"]="48 96 192 336"
#    ["Economy_sort"]="6 8 10 12"
)

# Define sequence lengths for each dataset
declare -A datasets_seq_len=(
    ["Agriculture"]="8"
    ["Climate"]="8"
    ["Economy"]="8"
    ["Energy"]="36"
    ["Environment"]="96"
    ["Health"]="36"
    ["Security"]="8"
    ["SocialGood"]="8"
    ["Traffic"]="8"
    ["new_metr"]="96"
    ["Economy_sort"]="8"
)

# Define label lengths for each dataset
declare -A datasets_label_len=(
    ["Agriculture"]="4"
    ["Climate"]="4"
    ["Economy"]="4"
    ["Energy"]="18"
    ["Environment"]="48"
    ["Health"]="18"
    ["Security"]="4"
    ["SocialGood"]="4"
    ["Traffic"]="4"
    ["new_metr"]="48"
    ["Economy_sort"]="4"
)

# Basic configuration (from second script)
all_models=("PatchTST_clip")
root_path=./data
current_dir=$(pwd)

# Fixed parameters (not tuned)
prior_weight=0.5
seeds=(2025)

# Hyperparameters to tune
batch_size_values=(32)

learning_rate_values=(0.00001)
learning_rate_clip_values=(0.0001)
learning_rate_corr_values=(0.001 0.0001 0.00001)

train_epochs_values=(10)
train_epochs_clip_values=(10)
train_epochs_corr_values=(10) #  40

correct_values=(1)
if_lradj_values=(1)
clip_t_values=(0.0)
dropout_values=(0.1)
decomp_w_values=(0.5)
# Main script name
MAIN_SCRIPT="run_clip.py"



# Initialize job counter
current_jobs=0

# Track all PIDs for better job management
declare -a job_pids=()

# Function to display running jobs
function show_running_jobs() {
    echo "Currently running jobs: $current_jobs out of maximum $MAX_PARALLEL"
    for pid in "${job_pids[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            echo "  - Job PID: $pid is still running"
        fi
    done
}

# Main hyperparameter tuning loops
for seed in "${seeds[@]}"
do
    for model_name in "${all_models[@]}"
    do
        for decomp_w in "${decomp_w_values[@]}"
        do
            # Loop through each dataset
            for dataset in "${!datasets[@]}"
            do
                # Get dataset-specific parameters
                pred_lens=(${datasets[$dataset]})
                seq_len=${datasets_seq_len[$dataset]}
                label_len=${datasets_label_len[$dataset]}

                data_path=${dataset}.csv
                model_id=$(basename ${root_path})

                echo "Processing dataset: $dataset"
                echo "  - Prediction lengths: ${pred_lens[@]}"
                echo "  - Sequence length: $seq_len"
                echo "  - Label length: $label_len"

                # Loop through each pred_len for this dataset
                for pred_len in ${pred_lens[@]}
                do
                    for batch_size in "${batch_size_values[@]}"
                    do
                        for correct in "${correct_values[@]}"
                        do
                            for learning_rate in "${learning_rate_values[@]}"
                            do
                                for learning_rate_clip in "${learning_rate_clip_values[@]}"
                                do
    #                                for learning_rate_corr in "${learning_rate_corr_values[@]}"
    #                                do
                                    learning_rate_corr=$learning_rate
                                    for train_epochs in "${train_epochs_values[@]}"
                                    do
                                        for train_epochs_clip in "${train_epochs_clip_values[@]}"
                                        do
                                            for train_epochs_corr in "${train_epochs_corr_values[@]}"
                                            do
                                                for if_lradj in "${if_lradj_values[@]}"
                                                do
                                                    for clip_t in "${clip_t_values[@]}"
                                                    do
                                                        for dropout in "${dropout_values[@]}"
                                                        do
                                                            # Construct a unique model ID and log file name
                                                            MODEL_ID="${model_name}_${dataset}_seed_${seed}_pred_${pred_len}_seq_${seq_len}_label_${label_len}_bs_${batch_size}_corr_${correct}_lr_${learning_rate}_lr_clip_${learning_rate_clip}_lr_corr_${learning_rate_corr}_e_${train_epochs}_e_clip_${train_epochs_clip}_e_corr_${train_epochs_corr}_if_lradj_${if_lradj}_clip_t_${clip_t}_dt_${dropout}_dw_${decomp_w}"
                                                            LOG_FILE="${log_path}/${MODEL_ID}.log"

                                                            if [[ -f "$LOG_FILE" ]]; then
                                                                echo "Output file $LOG_FILE already exists. Skipping task."
                                                                continue
                                                            fi

                                                            # Wait if we've reached the maximum number of parallel jobs
                                                            while [ $current_jobs -ge $MAX_PARALLEL ]; do
                                                                echo "Maximum parallel jobs ($MAX_PARALLEL) reached. Waiting for a job to complete..."
                                                                show_running_jobs
                                                                wait -n  # Wait for any job to finish

                                                                # Update job count by checking which jobs are still running
                                                                new_job_pids=()
                                                                for pid in "${job_pids[@]}"; do
                                                                    if kill -0 $pid 2>/dev/null; then
                                                                        new_job_pids+=($pid)
                                                                    fi
                                                                done
                                                                job_pids=("${new_job_pids[@]}")
                                                                current_jobs=${#job_pids[@]}
                                                            done

                                                            # Run the training command in the background
                                                            echo "Starting training: $model_name on $dataset with pred_len=$pred_len, seq_len=$seq_len, label_len=$label_len, batch_size=$batch_size, correct=$correct, lr=$learning_rate, lr_clip=$learning_rate_clip, lr_corr=$learning_rate_corr, e=$train_epochs, e_clip=$train_epochs_clip, e_corr=$train_epochs_corr, if_lradj=${if_lradj}, clip_t=${clip_t}, dt=${dropout}"

                                                            python -u $MAIN_SCRIPT \
                                                                --task_name long_term_forecast \
                                                                --is_training 1 \
                                                                --root_path $root_path \
                                                                --data_path $data_path \
                                                                --model_id $MODEL_ID \
                                                                --model $model_name \
                                                                --data custom \
                                                                --seq_len $seq_len \
                                                                --label_len $label_len \
                                                                --pred_len $pred_len \
                                                                --des Exp \
                                                                --seed $seed \
                                                                --save_name result_clip \
                                                                --multimodal 1 \
                                                                --correct $correct \
                                                                --tr_sea 1 \
                                                                --llm_model GPT2 \
                                                                --batch_size $batch_size \
                                                                --train_epochs $train_epochs \
                                                                --train_epochs_clip $train_epochs_clip \
                                                                --train_epochs_corr $train_epochs_corr \
                                                                --learning_rate $learning_rate \
                                                                --learning_rate_clip $learning_rate_clip \
                                                                --learning_rate_corr $learning_rate_corr \
                                                                --if_lradj $if_lradj \
                                                                --clip_t $clip_t \
                                                                --dropout $dropout \
                                                                --decomp_w $decomp_w \
                                                                --patience 5 > $LOG_FILE 2>&1 &

                                                            # Store the PID
                                                            pid=$!
                                                            job_pids+=($pid)
                                                            echo "Started process PID: $pid - Log file: $LOG_FILE"

                                                            current_jobs=${#job_pids[@]}
                                                            echo "Current running jobs: $current_jobs/$MAX_PARALLEL"

                                                            # Optional: Sleep briefly between job starts
                                                            # sleep 1
                                                        done
                                                    done
                                                done
                                            done
#                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

# Wait for all remaining background jobs to complete
echo "Waiting for all remaining jobs to complete..."
show_running_jobs
wait

echo "All training tasks have completed!"
echo "Check the logs directory for training logs."