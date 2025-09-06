#!/bin/bash
model_name=${SHARE_RES_DIR}/models/qwen/Qwen2.5-1.5B-Instruct
# model_name=${SHARE_RES_DIR}/models/deepseek/DeepSeek-R1-Distill-Qwen-7B


lr=1e-05
wd=0.01
warmup=0.05
subsample_ratio=1.0
# split=naive
task_name=ctrl_RE_id
master_port=29500
lr_scheduler_type=constant
bs=16
rr=0.0

while [[ $# -gt 0 ]]; do
    case $1 in
        --split) split="$2"; shift 2 ;;
        --rr) rr="$2"; shift 2;;
        --epochs) epochs="$2"; shift 2 ;;
        --bs) bs="$2"; shift 2 ;;
        --task_name) task_name="$2"; shift 2 ;;
        --gpu_ids) gpu_ids="$2"; shift 2 ;;
        --master_port) master_port="$2"; shift 2 ;;
        --lr_scheduler_type) lr_scheduler_type="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES=$gpu_ids
gpu_count=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")

per_device_train_batch_size=1
grad_acc=$((bs / $gpu_count / $per_device_train_batch_size))

pretty_name=${model_name##*/}
if [ "$subsample_ratio" = "1.0" ]; then
    run_name="${task_name}-${split}-lr${lr}-rr${rr}-epochs${epochs}-bs${bs}-wd${wd}-${lr_scheduler_type}-warmup${warmup}-${pretty_name}"
else
    run_name="scaling-subsample_ratio${subsample_ratio}-${task_name}-${split}-lr${lr}-rr${rr}-epochs${epochs}-bs${bs}-wd${wd}-${lr_scheduler_type}-warmup${warmup}-${pretty_name}"
fi
output_dir="ckpts/${run_name}"


# accelerate launch --config_file="default_config.yaml" \
#     --num_processes ${gpu_count} \
#     --main_process_port 0 \
torchrun --master_port=$master_port --nproc_per_node=$gpu_count \
    train.py \
    --model_name=$model_name \
    --block_size=2048 \
    --per_device_train_batch_size=$per_device_train_batch_size \
    --per_device_eval_batch_size=3 \
    --gradient_accumulation_steps=$grad_acc \
    --num_train_epochs=$epochs \
    --learning_rate=$lr \
    --rehersal_rate=$rr \
    --subsample_ratio=$subsample_ratio \
    --overwrite_output_dir=True \
    --task_name=$task_name \
    --split=$split \
    --logging_steps=1 \
    --run_name=$run_name \
    --bf16=True \
    --output_dir=$output_dir \
    --weight_decay=$wd \
    --warmup_ratio=$warmup \
    --do_eval=False \
    --save_strategy="no" \
    --lr_scheduler_type=$lr_scheduler_type \
    --log_level="info" \
    --fsdp="full_shard auto_wrap offload" \
    --fsdp_config="scripts/config/qwen_config.json"