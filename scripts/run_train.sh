split=meta_aug-one_stage-ice
rr=0.1 # 0.0, 0.1
gpu_ids=1
master_port=29501

for task_name in ctrl_RE_id ctrl_RE_ood_entity ctrl_RE_ood_relation ctrl_RE_ood_both; do
    bash scripts/train.sh --task_name $task_name --split $split --rr $rr --bs 16 --epochs 8 --gpu_ids $gpu_ids --master_port $master_port
    sleep 10
done