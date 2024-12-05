#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
model_name=${SHARE_RES_DIR}/models/llama3/hf/Meta-Llama-3-8B
gpu_count=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")

# bs=$gpu_count
bs=1
epochs=4
wd=1e-8
# lr=5e-06
# lr=5e-05

rr=0.0
warmup=0.1
subsample_ratio=1.0
task_name=musique
per_device_train_batch_size=1
grad_acc=$((bs / $gpu_count / $per_device_train_batch_size))


lr=1e-05


# for lr in 1e-05 # 2e-05 8e-06 5e-05 # 1e-05 3e-06 # 4e-06 2e-05  # 5e-07 5e-08 
# for example_id in 2hop__132710_120035 2hop__258019_119986 2hop__390772_565667 2hop__60060_25017 2hop__710977_25111 2hop__13778_15345 2hop__341498_76347 2hop__508013_351187 2hop__661591_13728 2hop__72949_9902
# do

# pretty_name=${model_name##*/}
# if [ "$subsample_ratio" = "1.0" ]; then
#     run_name="${task_name}-lr${lr}-rr${rr}-epochs${epochs}-bs${bs}-wd${wd}-warmup${warmup}-${pretty_name}"
# else
#     run_name="scaling-subsample_ratio${subsample_ratio}-${task_name}-lr${lr}-rr${rr}-epochs${epochs}-bs${bs}-wd${wd}-warmup${warmup}-${pretty_name}"
# fi
# output_dir="ckpts/${run_name}"

# # echo $pretty_name
# # exit
# export ACCELERATE_USE_FSDP=true
# # torchrun --nproc_per_node=$gpu_count
#     # --main_process_port=29500 \
#     # --main_process_port=0
#     # --main_process_port 0 \
# accelerate launch --config_file="default_config.yaml" \
#     train_musique_no_trainer.py \
#     --model_name=$model_name \
#     --block_size=512 \
#     --per_device_train_batch_size=${per_device_train_batch_size} \
#     --per_device_eval_batch_size=1 \
#     --gradient_accumulation_steps=$grad_acc \
#     --num_train_epochs=$epochs \
#     --learning_rate=$lr \
#     --rehersal_rate=$rr \
#     --subsample_ratio=$subsample_ratio \
#     --overwrite_output_dir=True \
#     --logging_steps=1 \
#     --run_name=$run_name \
#     --bf16=True \
#     --output_dir="${output_dir}-no-trainer" \
#     --dataloader_drop_last=False \
#     --weight_decay=$wd \
#     --warmup_ratio=$warmup \
#     --evaluation_strategy="epoch" \
#     --save_strategy="no" \
#     --lr_scheduler_type="constant" \
#     --log_level="info" \
#     --fsdp="full_shard auto_wrap" \
#     --task_name=$task_name \
#     --example_id=${example_id} # "2hop__132710_120035"
#     # --save_total_limit=1 \
#     # --load_best_model_at_end=True \
#     # --lr_scheduler_type="cosine" \
# done



task_name=musique_page

# for example_id in 2hop__132710_120035 2hop__258019_119986 2hop__390772_565667 2hop__60060_25017 2hop__710977_25111 
for example_id in 2hop__13778_15345 2hop__341498_76347 2hop__508013_351187 2hop__661591_13728 2hop__72949_9902
do

pretty_name=${model_name##*/}
if [ "$subsample_ratio" = "1.0" ]; then
    run_name="${task_name}-lr${lr}-rr${rr}-epochs${epochs}-bs${bs}-wd${wd}-warmup${warmup}-${pretty_name}"
else
    run_name="scaling-subsample_ratio${subsample_ratio}-${task_name}-lr${lr}-rr${rr}-epochs${epochs}-bs${bs}-wd${wd}-warmup${warmup}-${pretty_name}"
fi
output_dir="ckpts/${run_name}"

export ACCELERATE_USE_FSDP=true

accelerate launch --config_file="default_config.yaml" \
    --main_process_port 29600 \
    train_musique_no_trainer.py \
    --model_name=$model_name \
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
    --run_name=$run_name \
    --bf16=True \
    --output_dir="${output_dir}-no-trainer" \
    --dataloader_drop_last=False \
    --weight_decay=$wd \
    --warmup_ratio=$warmup \
    --evaluation_strategy="epoch" \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --log_level="info" \
    --fsdp="full_shard auto_wrap" \
    --task_name=$task_name \
    --example_id=${example_id} # "2hop__132710_120035"
    # --save_total_limit=1 \
    # --load_best_model_at_end=True \
    # --lr_scheduler_type="cosine" \
done



