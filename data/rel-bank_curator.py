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

relation_triplet_content_extractor = extractor.tag_content_extractor("relation_triplet")
relation_template_content_extractor = extractor.tag_content_extractor("relation_template")


class RelationBankGenerator(curator.LLM):
    def prompt(self, input: dict) -> str:
        """Generate a prompt for the subsubject generator."""
        return input["prompt"]

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        
        input["completion"] = response
        relation_triplets = relation_triplet_content_extractor(response)
        relation_templates = relation_template_content_extractor(response)
        relations = [
            {
                "relation_triplet": relation_triplet.strip(),
                "relation_template": relation_template.strip()
            }
            for relation_triplet, relation_template in zip(relation_triplets, relation_templates)
        ]
        input["relations"] = relations
        # input["relation_templates"] = relation_templates
        assert len(relations) == len(relation_templates)
        # import pdb; pdb.set_trace()
        return {**input}

model_name = "gpt-5"

rel_bank_generator = RelationBankGenerator(model_name=model_name)
fpath = "/home/zliu/zliu/Synthetic_Continued_Pretraining_leo/data/dataset/raw/4K_controlled_RE/train_relation_bank_construction.jsonl"
examples = io.load_jsonlines(fpath)
dataset = Dataset.from_list(examples[:])
dataset = rel_bank_generator(dataset)

fpath_wo_extention = io.remove_last_extension(fpath)
# import pdb; pdb.set_trace()
io.dump_jsonlines(dataset.to_list(), fpath_wo_extention + f"_generated({model_name}).jsonl")
# print()