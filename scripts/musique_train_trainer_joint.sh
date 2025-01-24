#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1 #0,1,2,3
model_name=${SHARE_RES_DIR}/models/llama3/hf/Meta-Llama-3-8B
# model_name=${SHARE_RES_DIR}/models/qwen/Qwen1.5-1.8B
gpu_count=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")

# bs=$gpu_count
bs=4
epochs=4
wd=1e-8
lr=1e-05

# lr=5e-05

rr=0.1
warmup=0.1
subsample_ratio=1.0

per_device_train_batch_size=1
grad_acc=$((bs / gpu_count / per_device_train_batch_size))

max_grad_norm=1.0

task_name=musique_joint
# task_name=musique_page_joint

lr_scheduler_type=cosine

# for max_grad_norm in 0.0 0.5 1.0
# do
# for lr in 1e-05 1e-06 1e-04 1e-07 1e-08
# do
for task_name in musique_joint # musique_page_joint
do
for lr in 5e-08 1e-08 # 5e-05 5e-06 5e-07 
do
for example_id in 50instances
do

pretty_name=${model_name##*/}
if [ "$subsample_ratio" = "1.0" ]; then
    run_name="${task_name}-lr${lr}-rr${rr}-epochs${epochs}-bs${bs}-wd${wd}-warmup${warmup}-norm${max_grad_norm}-${lr_scheduler_type}-ngpu${gpu_count}-${pretty_name}"
else
    run_name="scaling-subsample_ratio${subsample_ratio}-${task_name}-lr${lr}-rr${rr}-epochs${epochs}-bs${bs}-wd${wd}-warmup${warmup}-${pretty_name}"
fi
output_dir="ckpts/${run_name}"

export ACCELERATE_USE_FSDP=true
export CUDA_LAUNCH_BLOCKING=1
echo "Example ID: ${example_id}"

accelerate launch --config_file="default_config.yaml" \
    --main_process_port 29500 \
    --num_processes ${gpu_count} \
    train_musique_joint.py \
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
    --output_dir="${output_dir}-trainer" \
    --max_grad_norm=${max_grad_norm} \
    --dataloader_drop_last=False \
    --weight_decay=$wd \
    --warmup_ratio=$warmup \
    --evaluation_strategy="epoch" \
    --save_strategy="no" \
    --lr_scheduler_type=${lr_scheduler_type} \
    --log_level="info" \
    --report_to="none" \
    --fsdp="full_shard auto_wrap" \
    --task_name=$task_name \
    --eval_on_start=True \
    --example_id=${example_id} \
    # --save_total_limit=1 \
    # --load_best_model_at_end=True \
    # --lr_scheduler_type="cosine" \
# python eval_musique.py --task_name=$task_name --example_id=${example_id} --model_name="${model_name}" --output_dir="${output_dir}-trainer"

python eval_musique_joint.py --data_dir /u/zliu/datastor1/KE-by-CP/data/musique_c_small --data_file examples-page.jsonl --example_id=${example_id} --model_name="${model_name}" --output_dir="${output_dir}-trainer"
# e.g. :
# CUDA_VISIBLE_DEVICES=7 python eval_musique_joint.py --data_dir /u/zliu/datastor1/KE-by-CP/data/musique_c_small --data_file examples-page.jsonl --model_name=${SHARE_RES_DIR}/models/llama3/hf/Meta-Llama-3-8B --output_dir=/u/zliu/datastor1/Synthetic_Continued_Pretraining/ckpts/musique_page_joint-lr1e-05-rr0.1-epochs4-bs4-wd1e-8-warmup0.1-norm1.0-cosine-ngpu4-Meta-Llama-3-8B-trainer
rm -rf "${output_dir}-trainer/tmp_ckpt"

done
done
done