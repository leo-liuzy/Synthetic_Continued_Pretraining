


# model_name_or_path="/u/zliu/datastor1/shared_resources/models/qwen/Qwen2.5-1.5B-Instruct"
model_name_or_path="ckpts/ctrl_RE_id-naive-lr1e-05-rr0.0-epochs8-bs16-wd0.01-constant-warmup0.05-Qwen2.5-1.5B-Instruct"
# model_name_or_path="ckpts/ctrl_RE_id-entigraph-lr1e-05-rr0.0-epochs2-bs16-wd0.01-cosine-warmup0.05-Qwen2.5-1.5B-Instruct"
# model_name_or_path="ckpts/controlled_RE_id-entigraph-lr1e-05-rr0.1-epochs2-bs16-wd0.01-cosine-warmup0.05-Qwen2.5-1.5B-Instruct"


# model_name_or_path="/u/zliu/datastor1/shared_resources/models/deepseek/DeepSeek-R1-Distill-Qwen-7B"


rr=0.0

test_set_pairs=(
    "ctrl_RE_id test_id_sample"
    "ctrl_RE_ood_both test_ood_both_sample"
    "ctrl_RE_ood_entity test_ood_entity_sample"
    "ctrl_RE_ood_relation test_ood_relation_sample"
)

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name_or_path) model_name_or_path="$2"; shift 2 ;;
        --gpu_id) gpu_id="$2"; shift 2 ;;
        --cpt_data_choice) cpt_data_choice="$2"; shift 2 ;;
        --rr) rr="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES=${gpu_id}


# cpt_data_choice = "naive" "meta_aug-one_stage-naive" "meta_aug-one_stage-ice" 

eval_data_name="controlled_RE_efficacy"

# for rr in 0.0 # 0.1; 
# do

# for cpt_data_choice in "naive" # "meta_aug-one_stage-naive" "meta_aug-one_stage-ice"; do
# do

for pair in "${test_set_pairs[@]}"; do
    read -r task_name test_set_choice <<< "$pair"
    model_name_or_path=ckpts/${task_name}-${cpt_data_choice}-lr1e-05-rr${rr}-epochs8-bs16-wd0.01-constant-warmup0.05-Qwen2.5-1.5B-Instruct
    if [ "$test_set_choice" != "ctrl_RE_id" ]; then
        python query_vllm.py --model-name-or-path $model_name_or_path --eval-data-name controlled_RE_efficacy --max-tokens 1024 --test-set-choice $test_set_choice # --overwrite
    else
        for eval_data_name in "all"; do
            python query_vllm.py --model-name-or-path $model_name_or_path --eval-data-name $eval_data_name --max-tokens 1024 --test-set-choice $test_set_choice # --overwrite
        done
    fi
done
# done
# done

# test_set_choice="test_id_sample"
# for eval_data_name in "controlled_RE_efficacy" # "all" "controlled_RE_specificity" # "controlled_RE_efficacy" # "controlled_RE_specificity" "mmlu_0shot_cot"
# do
#     python query_vllm.py --model-name-or-path $model_name_or_path --eval-data-name $eval_data_name --max-tokens 1024 --test-set-choice $test_set_choice 
# done
