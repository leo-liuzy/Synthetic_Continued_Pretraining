export PYTHONPATH=/data/users/zliu/Synthetic_Continued_Pretraining:$PYTHONPATH

python data/entigraph.py 0 --dataset jd-vance --sample_triplet_ratio=0.2
# python data/tokenize_entigraph.py --dataset jd-vance --sample_triplet_ratio=0.2