export PYTHONPATH=/u/zliu/datastor1/Synthetic_Continued_Pretraining:$PYTHONPATH

for i in {0..9}
do
    # python data/entigraph.py $i --dataset musique --max_n_entity_allowed 18 &
done

# python data/tokenize_entigraph.py --dataset KE-by-CP
# python data/tokenize_entigraph.py --dataset KE-by-CP --no_triplet