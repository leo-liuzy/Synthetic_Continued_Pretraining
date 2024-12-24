import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from typing import List
import numpy as np
from transformers import AutoTokenizer
import random
import glob
from tqdm import tqdm
from utils.io_utils import jload
import argparse


def get_tokenizer(tokenizer_model_name: str) -> AutoTokenizer:
    # loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name, use_fast=True)
    tokenizer.model_max_length = 2**20  # this is to hide the token_len>128K wraning
    return tokenizer


def tokenize_list(args, text_list: List[str]) -> List[int]:
    """
    Tokenize the text and return the tokenized text
    """
    random.shuffle(text_list)
    tokenizer = get_tokenizer(args.model_name_or_path)
    all_ids = []
    for text in tqdm(text_list):
        if text:
            ids = tokenizer.encode(text)  # add_special_tokens=True to add BOS token
            ids.append(tokenizer.eos_token_id)  # add the end of text token
            all_ids.extend(ids)
    return all_ids


def write_to_memmap_single(ids: List[int], filename: str):
    # filename = f'data/dataset/bins/{filename}'
    print(f"Writing to {filename} with length {len(ids)}")
    dtype = np.int32
    ids_arr = np.array(ids, dtype=dtype)
    arr_len = len(ids_arr)
    arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
    arr[:] = ids_arr
    arr.flush()


def _glob_all_json(dir_name: str) -> List[str]:
    return glob.glob(f"{dir_name}/*.json") + glob.glob(f"{dir_name}/.*.json")


def tokenize_quality_graph(args):
    # quality = _get_quality_graph(args)
    bin_dirname = f"data/dataset/bins/{args.dataset}"
    os.makedirs(bin_dirname, exist_ok=True)

    dir_name = f"data/dataset/raw/{args.dataset}"
    files = _glob_all_json(dir_name)

    for file in files:
        assert file.endswith(".json")
        file_id = os.path.basename(file)[:-5]
        content = jload(file)

        write_to_memmap_single(tokenize_list(args, content[1:]), f"{bin_dirname}/{file_id.split(' ')[0]}.bin")


if __name__ == "__main__":
    # Writing to data/dataset/bins/quality_all-graphgpt-4-turbo.bin with length 599385906 (599M)
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="musique_page")

    parser.add_argument(
        "--model_name_or_path", type=str, default=f"{os.environ['SHARE_RES_DIR']}/models/llama3/hf/Meta-Llama-3-8B/"
    )
    args = parser.parse_args()

    tokenize_quality_graph(args)
