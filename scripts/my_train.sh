#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
# model_name=${SHARE_RES_DIR}/models/llama3/hf/Meta-Llama-3-8B
model_name=${SHARE_RES_DIR}/models/qwen/Qwen1.5-1.8B
gpu_count=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")



bs=4
epochs=2
wd=0.01
lr=5e-06
# lr=5e-05

rr=0.1
warmup=0.05
subsample_ratio=1.0
task_name=jd-vance
per_device_train_batch_size=2
grad_acc=$((bs / $gpu_count / $per_device_train_batch_size))

pretty_name=${model_name##*/}
if [ "$subsample_ratio" = "1.0" ]; then
    run_name="${task_name}-lr${lr}-rr${rr}-epochs${epochs}-bs${bs}-wd${wd}-warmup${warmup}-${pretty_name}"
else
    run_name="scaling-subsample_ratio${subsample_ratio}-${task_name}-lr${lr}-rr${rr}-epochs${epochs}-bs${bs}-wd${wd}-warmup${warmup}-${pretty_name}"
fi
output_dir="ckpts/${run_name}"

# echo $pretty_name
# exit
export ACCELERATE_USE_FSDP=true
# torchrun --nproc_per_node=$gpu_count
    # --main_process_port=29500 \
accelerate launch --config_file="default_config.yaml" \
    train.py \
    --model_name=$model_name \
    --block_size=2048 \
    --per_device_train_batch_size=${per_device_train_batch_size} \
    --per_device_eval_batch_size=3 \
    --gradient_accumulation_steps=$grad_acc \
    --num_train_epochs=$epochs \
    --learning_rate=$lr \
    --rehersal_rate=$rr \
    --subsample_ratio=$subsample_ratio \
    --overwrite_output_dir=True \
    --task_name=$task_name \
    --logging_steps=1 \
    --run_name=$run_name \
    --bf16=True \
    --output_dir=$output_dir \
    --weight_decay=$wd \
    --warmup_ratio=$warmup \
    --evaluation_strategy="no" \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --log_level="info" \
    --report_to="none" \
    --fsdp="full_shard auto_wrap" \
    --no_pair \
    --no_triplet \
    --fsdp_config="default_config.yaml" \
    --max_steps=1
    # --sample_triplet_ratio=0.2 \
    # --trimE \
    # --use_peft=True
    
    
# 03:48


# --fsdp="hybrid_shard auto_wrap" \