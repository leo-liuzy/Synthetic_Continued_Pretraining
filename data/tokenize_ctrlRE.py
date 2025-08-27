from typing import List
import numpy as np
from transformers import AutoTokenizer
import random
import glob
from tqdm import tqdm
from utils.io_utils import jload

def _glob_all_json(dir_name: str) -> List[str]:
    return glob.glob(f'{dir_name}/*.json') + glob.glob(f'{dir_name}/.*.json')

def _get_quality_graph(dir_name: str) -> List[str]:
    files = _glob_all_json(dir_name)
    result = []
    for file in files:
        content = jload(file)
        result.extend(content[1:])
    return result

def get_tokenizer(tokenizer_model_name: str)-> AutoTokenizer:
    # loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name, use_fast=True)
    tokenizer.model_max_length=2**20 # this is to hide the token_len>128K wraning
    return tokenizer

def tokenize_list(text_list: List[str], tokenizer_name="meta-llama/Meta-Llama-3-8B") -> List[int]:
    """
    Tokenize the text and return the tokenized text
    """
    random.shuffle(text_list)
    tokenizer = get_tokenizer(tokenizer_name)
    all_ids = []
    for text in tqdm(text_list):
        if text:
            ids = tokenizer.encode(text) # add_special_tokens=True to add BOS token
            ids.append(tokenizer.eos_token_id) # add the end of text token
            all_ids.extend(ids)
    return all_ids

def write_to_memmap_single(ids: List[int], filename: str, dir_path="data/dataset/bins"):
    filename = f'{dir_path}/{filename}'
    print(f'Writing to {filename} with length {len(ids)}')
    dtype = np.int32
    ids_arr = np.array(ids, dtype=dtype)
    arr_len = len(ids_arr)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    arr[:] = ids_arr
    arr.flush()
    
if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--cpt_text_construct', type=str, default='entigraph',
                        help='Source of text data (e.g. entigraph, naive)')
    parser.add_argument('--test_data_name', type=str, default='test_id_sample',
                        help='Tokenizer name')
    parser.add_argument('--tokenizer_name', type=str, default='Qwen/Qwen2.5-1.5B-Instruct',
                        help='Tokenizer name')
    
    args = parser.parse_args()
    cpt_text_construct = args.cpt_text_construct
    test_data_name = args.test_data_name
    tokenizer_name = args.tokenizer_name

    PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # import pdb; pdb.set_trace()
    
    corpus_lst = _get_quality_graph(f"{PROJ_DIR}/data/dataset/raw/4K_controlled_RE/{test_data_name}/{cpt_text_construct}")

    tokenized_corpus = tokenize_list(corpus_lst, tokenizer_name=tokenizer_name)

    write_to_memmap_single(tokenized_corpus, filename=f"4K_controlled_RE-{test_data_name}-{cpt_text_construct}-{os.path.basename(tokenizer_name)}.bin", dir_path = f"{PROJ_DIR}/data/dataset/bins")