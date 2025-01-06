#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
# model_name=${SHARE_RES_DIR}/models/llama3/hf/Meta-Llama-3-8B
model_name=${SHARE_RES_DIR}/models/qwen/Qwen1.5-1.8B
gpu_count=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")

bs=4
# bs=1
epochs=1
wd=1e-8
# lr=5e-06
# lr=1e-05
lr=0

rr=0.1
warmup=0.1
subsample_ratio=1.0

per_device_train_batch_size=1
grad_acc=$((bs / gpu_count / per_device_train_batch_size))

max_grad_norm=1.0


task_name=musique_page

lr_scheduler_type=cosine

for example_id in 2hop__132710_120035 # 2hop__258019_119986 2hop__390772_565667 2hop__60060_25017 2hop__710977_25111 2hop__13778_15345 2hop__341498_76347 2hop__508013_351187 2hop__661591_13728 2hop__72949_9902
do

pretty_name=${model_name##*/}
if [ "$subsample_ratio" = "1.0" ]; then
    run_name="${task_name}-lr${lr}-rr${rr}-epochs${epochs}-bs${bs}-wd${wd}-warmup${warmup}-norm${max_grad_norm}-${lr_scheduler_type}-ngpu${gpu_count}-${pretty_name}"
else
    run_name="scaling-subsample_ratio${subsample_ratio}-${task_name}-lr${lr}-rr${rr}-epochs${epochs}-bs${bs}-wd${wd}-warmup${warmup}-${pretty_name}"
fi
output_dir="ckpts/${run_name}_debug"

export ACCELERATE_USE_FSDP=true

accelerate launch --config_file="default_config.yaml" \
    --main_process_port 29500 \
    --num_processes "${gpu_count}" \
    train_musique_no_trainer_compact.py \
    --model_name="$model_name" \
    --block_size=512 \
    --per_device_train_batch_size=${per_device_train_batch_size} \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=$grad_acc \
    --num_train_epochs=$epochs \
    --learning_rate=$lr \
    --rehersal_rate=$rr \
    --subsample_ratio=$subsample_ratio \
    --overwrite_output_dir=True \
    --logging_steps=1 \
    --run_name="$run_name" \
    --bf16=False \
    --output_dir="${output_dir}-no-trainer" \
    --max_grad_norm=${max_grad_norm} \
    --dataloader_drop_last=False \
    --weight_decay=$wd \
    --warmup_ratio=$warmup \
    --evaluation_strategy="epoch" \
    --save_strategy="no" \
    --lr_scheduler_type=${lr_scheduler_type} \
    --log_level="info" \
    --fsdp="full_shard auto_wrap" \
    --task_name=$task_name \
    --example_id=${example_id} # "2hop__132710_120035"
    # --save_total_limit=1 \
    # --load_best_model_at_end=True \
    # --lr_scheduler_type="cosine" \
done



