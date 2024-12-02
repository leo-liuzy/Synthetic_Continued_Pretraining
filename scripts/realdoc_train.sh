#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
model_name=/home/zliu/shared_resources/models/llama3/hf/Meta-Llama-3-8B
gpu_count=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")

bs=1
epochs=4
wd=1e-8
# lr=5e-06
# lr=5e-05

rr=0.0
warmup=0.1
subsample_ratio=1.0
task_name=real-jd-vance
per_device_train_batch_size=1
grad_acc=$((bs / $gpu_count / $per_device_train_batch_size))

# lr=1e-05
for lr in 2e-05 8e-06 5e-05 # 1e-05 3e-06 # 4e-06 2e-05  # 5e-07 5e-08 
do

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
    # --main_process_port=0
accelerate launch --config_file="default_config.yaml" \
    train.py \
    --model_name=$model_name \
    --block_size=512 \
    --per_device_train_batch_size=${per_device_train_batch_size} \
    --per_device_eval_batch_size=3 \
    --gradient_accumulation_steps=$grad_acc \
    --num_train_epochs=$epochs \
    --learning_rate=$lr \
    --rehersal_rate=$rr \
    --subsample_ratio=$subsample_ratio \
    --overwrite_output_dir=True \
    --logging_steps=1 \
    --run_name=$run_name \
    --bf16=True \
    --output_dir=$output_dir \
    --dataloader_drop_last=False \
    --weight_decay=$wd \
    --warmup_ratio=$warmup \
    --evaluation_strategy="epoch" \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --log_level="info" \
    --fsdp="full_shard auto_wrap" \
    --task_name=$task_name \
    --train_split=12doc \
    --valid_split=valid
    # --save_total_limit=1 \
    # --load_best_model_at_end=True \
done    
    # --sample_triplet_ratio=0.2 \
    # --trimE \
    # --use_peft=True
    
    
    # --fsdp_config="default_config.yaml"
# 03:48


# --fsdp="hybrid_shard auto_wrap" \