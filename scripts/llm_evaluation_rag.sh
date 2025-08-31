

for test_set_choice in test_id_sample test_ood_entity_sample test_ood_relation_sample test_ood_both_sample; do

CUDA_VISIBLE_DEVICES=1 python query_vllm_rag.py --model-name-or-path /home/zliu/shared_resources/models/qwen/Qwen2.5-1.5B-Instruct --eval-data-name controlled_RE_efficacy --max-tokens 1024 --test-set-choice $test_set_choice --rag-choice top1

done