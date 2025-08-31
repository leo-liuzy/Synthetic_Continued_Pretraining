test_data_name="test_id_sample"
# test_data_name="test_ood_relation_sample"
# test_data_name="test_ood_entity_sample"
# test_data_name="test_ood_both_sample"
#  "active_reading-task_specific" "active_reading-task_agnostic"
for cpt_text_construct in "active_reading-task_agnostic-task_specific"; do
# for cpt_text_construct in meta_aug-one_stage-naive meta_aug-one_stage-ice meta_aug-two_stage-naive meta_aug-two_stage-ice; do

python data/tokenize_ctrlRE.py --cpt_text_construct ${cpt_text_construct} --test_data_name ${test_data_name} 

done