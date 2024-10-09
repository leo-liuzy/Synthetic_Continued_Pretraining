export PYTHONPATH=/data/users/zliu/Synthetic_Continued_Pretraining:$PYTHONPATH

# python data/entigraph.py 0 --dataset KE-by-CP
python data/tokenize_entigraph.py --dataset KE-by-CP
python data/tokenize_entigraph.py --dataset KE-by-CP --no_triplet