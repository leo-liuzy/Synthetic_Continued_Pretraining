#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1 #0,1,2,3
# model_name=${SHARE_RES_DIR}/models/llama3/hf/Meta-Llama-3-8B
# model_name=${SHARE_RES_DIR}/models/llama3/hf/Llama-3.2-3B
model_name=${SHARE_RES_DIR}/models/llama3/hf/Llama-3.2-1B
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


task_name=musique_page
single_doc=$2
multi_edit=$3

final_task_name="${task_name}"

if [ "$single_doc" = "True" ]; then
    final_task_name="${final_task_name}_single"
else
    final_task_name="${final_task_name}_two"
fi

if [ "$multi_edit" = "False" ]; then
    final_task_name="${final_task_name}_single"
else
    final_task_name="${final_task_name}_multi"
fi
echo "Task Name: ${final_task_name}"

lr_scheduler_type=cosine

# for max_grad_norm in 0.0 0.5 1.0
# do
for lr in 1e-04 1e-06 5e-06 5e-05 1e-05 
do
# 2hop__132710_120035 
echo $lr
# for example_id in 2hop__258019_119986 2hop__390772_565667 2hop__60060_25017 2hop__710977_25111 2hop__13778_15345 2hop__341498_76347 2hop__508013_351187 2hop__661591_13728 2hop__72949_9902 2hop__132710_120035 # 
for example_id in 2hop__258019_119986 2hop__390772_565667 2hop__60060_25017 2hop__710977_25111 2hop__13778_15345 2hop__341498_76347 2hop__508013_351187 2hop__661591_13728 2hop__72949_9902 2hop__132710_120035 2hop__628385_161358 2hop__3257_2998 2hop__317528_774871 2hop__85865_86706 2hop__54758_446818 2hop__647869_2702 2hop__159827_9449 2hop__546986_565529 2hop__53030_79070 2hop__257846_500443 2hop__132590_663762 2hop__616216_8600 2hop__154226_727337 2hop__532353_58115 2hop__597354_86295 2hop__194896_77553 2hop__261004_259429 2hop__153274_49441 2hop__428289_24352 2hop__489969_44637 2hop__177017_74276 2hop__18025_34452 2hop__129608_112624 2hop__813239_161698 2hop__291833_3814 2hop__132018_91253 2hop__142671_126711 2hop__455016_823618 2hop__131724_58935 2hop__553369_872 2hop__658785_8607 2hop__831373_162428 2hop__72949_29454 2hop__269683_467995 2hop__806470_84477 2hop__35466_88461 2hop__836_919 2hop__711689_162428 2hop__454283_92444 2hop__251426_55948
do

pretty_name=${model_name##*/}
if [ "$subsample_ratio" = "1.0" ]; then
    run_name="${final_task_name}-lr${lr}-rr${rr}-epochs${epochs}-bs${bs}-wd${wd}-warmup${warmup}-norm${max_grad_norm}-${lr_scheduler_type}-ngpu${gpu_count}-${pretty_name}"
else
    run_name="scaling-subsample_ratio${subsample_ratio}-${final_task_name}-lr${lr}-rr${rr}-epochs${epochs}-bs${bs}-wd${wd}-warmup${warmup}-${pretty_name}"
fi
output_dir="ckpts/${run_name}"

if [ -f "${output_dir}-trainer/inference_results/${example_id}_inferencer_results.xlsx" ]; then
    continue
fi

export ACCELERATE_USE_FSDP=true
export CUDA_LAUNCH_BLOCKING=1
echo "Example ID: ${example_id}"


accelerate launch --config_file="default_config.yaml" \
    --main_process_port 29500 \
    --num_processes ${gpu_count} \
    train_musique.py \
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
python eval_musique.py --task_name=$task_name --example_id=${example_id} --model_name="${model_name}" --output_dir="${output_dir}-trainer"

# echo "Removing checkpoint"
rm -rf "${output_dir}-trainer/tmp_ckpt"
done
echo 
done
# done