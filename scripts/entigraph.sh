export PYTHONPATH=/u/zliu/datastor1/Synthetic_Continued_Pretraining:$PYTHONPATH

for i in {0..9}
do
    python data/entigraph.py $i --dataset musique --max_pair_entity_allowed 16 --max_triplet_entity_allowed 8 & 
    # python data/entigraph.py $i --dataset musique_single --max_triplet_entity_allowed 8 &
done

# python data/tokenize_entigraph.py --dataset KE-by-CP
# python data/tokenize_entigraph.py --dataset KE-by-CP --no_triplet