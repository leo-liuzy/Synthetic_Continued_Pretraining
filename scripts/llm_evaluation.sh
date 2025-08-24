
export CUDA_VISIBLE_DEVICES=1

# model_name_or_path="/u/zliu/datastor1/shared_resources/models/qwen/Qwen2.5-1.5B-Instruct"
# model_name_or_path="ckpts/controlled_RE_id-naive-lr1e-05-rr0.0-epochs2-bs16-wd0.01-constant-warmup0.05-Qwen2.5-1.5B-Instruct"
# model_name_or_path="ckpts/controlled_RE_id-entigraph-lr1e-05-rr0.0-epochs2-bs16-wd0.01-cosine-warmup0.05-Qwen2.5-1.5B-Instruct"
# model_name_or_path="ckpts/controlled_RE_id-entigraph-lr1e-05-rr0.1-epochs2-bs16-wd0.01-cosine-warmup0.05-Qwen2.5-1.5B-Instruct"


model_name_or_path="/u/zliu/datastor1/shared_resources/models/deepseek/DeepSeek-R1-Distill-Qwen-7B"

for eval_data_name in "controlled_RE_efficacy" "controlled_RE_specificity" # "controlled_RE_efficacy" # "controlled_RE_specificity" "mmlu_0shot_cot"
do
    python query_vllm.py --model-name-or-path $model_name_or_path --eval-data-name $eval_data_name --max-tokens 16384 --overwrite
done
