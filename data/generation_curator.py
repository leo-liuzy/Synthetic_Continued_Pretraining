from datasets import load_dataset
from tqdm import tqdm
from knowledge_propagation.utils import vars, io, extractor
from scipy.stats import describe
from typing import List, Dict
import re
from copy import deepcopy
import pandas as pd

from bespokelabs import curator
from datasets import Dataset


class Generator(curator.LLM):
    def prompt(self, input: dict) -> str:
        """Generate a prompt for the subsubject generator."""
        return input["prompt"]

    def parse(self, input: dict, response: str) -> dict:
        """
        Parse the model response along with the input to the model into the desired output format.
        """
        input["completion"] = response
        return {**input}

# model_name = "gpt-5"
model_name = "gpt-5-nano"

rel_bank_generator = Generator(model_name=model_name)
fpath = "/u/zliu/datastor1/Synthetic_Continued_Pretraining/data/dataset/raw/4K_controlled_RE/test_id_active_reading_strategy_construction.jsonl"
examples = io.load_jsonlines(fpath)
dataset = Dataset.from_list(examples[:])
dataset = rel_bank_generator(dataset)

fpath_wo_extention = io.remove_last_extension(fpath)
io.dump_jsonlines(dataset.to_list(), fpath_wo_extention + f"_generated({model_name}).jsonl")