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


class EntiGraphGenerator(curator.LLM):
    def prompt(self, input: dict) -> str:
        """Generate a prompt for the subsubject generator."""
        return input["prompt"]

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        
        input["completion"] = response
        return {**input}

llm_judge = EntiGraphGenerator(model_name="gpt-4-turbo")
fpath = "/home/zliu/zliu/Synthetic_Continued_Pretraining_leo/data/dataset/raw/4K_controlled_RE/test_id_sample_curator_prompt.xlsx"
# fpath = "/u/zliu/datastor1/mend/exp_output/eos-sft_musique_propagator_text_hidden_w-atomq/musique/mend_eval_loss=clm_input=hidden_n=1000_prompt=no_w-gen_wo-icl_spec.xlsx"
df = pd.read_excel(fpath)
dataset = Dataset.from_pandas(df)
dataset = llm_judge(dataset)

fpath_wo_extention = io.remove_last_extension(fpath)

dataset.to_pandas().to_excel(fpath_wo_extention + "_generated.xlsx", index=False)
print()